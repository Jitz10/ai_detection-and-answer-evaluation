from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pymongo import MongoClient
import base64
import io

app = Flask(__name__)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# MongoDB setup
mongo_client = MongoClient(os.getenv("mongo_client"))
db = mongo_client['fosip']
tests_collection = db['tests']

def get_pdf_text(pdf_data):
    """Extract text from PDF data"""
    pdf_file = io.BytesIO(pdf_data)
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def create_vector_store(text_chunks, test_id):
    """Create and save vector store"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(f"faiss_index_{test_id}")
    return True

def get_conversational_chain():
    """Create the evaluation chain"""
    prompt_template = """
    You are an evaluator responsible for assessing a user answer based on the provided question, context, and evaluation rubrics. Your goal is to provide an accurate score and explanation based on the rubrics and the information available.

    Instructions:
    - Carefully read the context to determine if it contains sufficient information to evaluate the user's answer.
    - If the context lacks adequate information to evaluate the answer, respond with: "Answer is not available in the context." and do not proceed further.
    - If the context is sufficient, assign a score from 0 to 10 at the beginning of your response (formatted as the first two characters). This score should be based on the rubrics and should be followed by a clear, detailed explanation justifying the score.
    - Highlight strengths and weaknesses as they relate to the rubrics.

    Components:

    Context:
    {context}

    Rubrics for Evaluation:
    {rubrics}

    Question:
    {question}

    User Answer:
    {user_answer}

    Your Evaluation:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "rubrics", "question", "user_answer"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

@app.route('/process_test', methods=['POST'])
def process_test():
    """Endpoint to process a new test PDF and create vector store"""
    try:
        data = request.json
        test_id = data.get('test_id')
        pdf_data = base64.b64decode(data.get('pdf_data'))
        
        # Extract text and create vector store
        raw_text = get_pdf_text(pdf_data)
        text_chunks = get_text_chunks(raw_text)
        create_vector_store(text_chunks, test_id)
        
        # Store test information in MongoDB
        tests_collection.update_one(
            {'test_id': test_id},
            {'$set': {
                'status': 'processed',
                'vector_store_path': f"faiss_index_{test_id}"
            }},
            upsert=True
        )
        
        return jsonify({
            'status': 'success',
            'message': f'Test {test_id} processed successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/evaluate_answer', methods=['POST'])
def evaluate_answer():
    """Endpoint to evaluate a user's answer"""
    try:
        data = request.json
        test_id = data.get('test_id')
        question = data.get('question')
        rubrics = data.get('rubrics')
        user_answer = data.get('user_answer')
        
        # Load vector store for the test
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local(
            f"faiss_index_{test_id}", 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Get relevant documents
        docs = vector_store.similarity_search(question)
        
        # Get evaluation chain
        chain = get_conversational_chain()
        
        # Get initial evaluation
        response = chain(
            {
                "input_documents": docs,
                "question": question,
                "rubrics": rubrics,
                "user_answer": user_answer
            },
            return_only_outputs=True
        )
        
        output_text = response["output_text"]
        
        # If context is insufficient, use Gemini directly
        if "Answer is not available in the context." in output_text:
            model = genai.GenerativeModel("gemini-1.5-flash")
            query = f'''
                You are an evaluator responsible for assessing a provided answer for evaluation.
                Your task is to generate an informed evaluation based on the provided question,
                answer, and rubrics.

                Rubrics for Evaluation:
                {rubrics}

                Question:
                {question}

                User Answer:
                {user_answer}

                Evaluation:
            '''
            
            genai_response = model.generate_content(query)
            output_text = genai_response.text
        
        return jsonify({
            'status': 'success',
            'evaluation': output_text
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)