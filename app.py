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
from io import BytesIO
app = Flask(__name__)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# MongoDB setup
mongo_client = MongoClient(os.getenv("mongo_client"))
db = mongo_client['fosip']
tests_collection = db['tests']


# def get_pdf_text(pdf_data):
#     """Extract text from PDF data"""
#     pdf_file = io.BytesIO(pdf_data)
#     pdf_reader = PdfReader(pdf_file)
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     return text

def get_text_from_binary_pdfs(pdf_binaries):
    """
    Extract text from a list of binary PDF data.

    Args:
        pdf_binaries (list): A list of binary data representing PDF files.

    Returns:
        str: Concatenated text from all the PDFs.
    """
    text = ""
    for binary_pdf in pdf_binaries:
        pdf_stream = BytesIO(binary_pdf)  # Wrap binary data in a BytesIO stream
        pdf_reader = PdfReader(pdf_stream)  # Read the PDF from the binary stream
        for page in pdf_reader.pages:
            text += page.extract_text()  # Extract text from each page
    return text


def get_text_chunks(text):
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


def create_vector_store(text_chunks, test_name):
    """Create and save vector store in the 'vector_databases' directory."""
    # Define the directory path
    vector_dir = "vector_databases"
    
    # Ensure the directory exists
    if not os.path.exists(vector_dir):
        os.makedirs(vector_dir)  # Create the directory if it doesn't exist
    
    # Define the full path for saving the vector store
    vector_path = os.path.join(vector_dir, f"faiss_index_{test_name}")
    
    # Create embeddings and the FAISS vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    # Save the vector store
    vector_store.save_local(vector_path)
    return True

def get_conversational_chain():
    """Create the evaluation chain"""
    prompt_template = """
    You are an evaluator responsible for assessing a user answer based on the provided question and context. Your goal is to provide an accurate score and explanation based on the information available.

    Instructions:
    - Carefully read the context to determine if it contains sufficient information to evaluate the user's answer.
    - If the context lacks adequate information to evaluate the answer, respond with: "Answer is not available in the context." and do not proceed further.
    - If the context is sufficient, assign a score from 0 to 10 at the beginning of your response (formatted as the first two characters). This score should be based on the information available and should be followed by a clear, detailed explanation justifying the score.
    - Highlight strengths and weaknesses as they relate to the user's answer.

    Components:

    Context:
    {context}

    Question:
    {question}

    User Answer:
    {user_answer}

    Your Evaluation:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "user_answer"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

@app.route('/',methods=['GET'])
def tp():
    return "Hello world"



@app.route('/process_test', methods=['POST'])
def process_test():
    """Endpoint to process a test PDF and create a vector store."""
    try:
        data = request.json
        test_name = data.get('test_name')
        if not test_name:
            return jsonify({
                'status': 'error',
                'message': 'Test name is required.'
            }), 400
        
        # Fetch test document from MongoDB
        test_document = tests_collection.find_one({"name": test_name})
        if not test_document or 'pdf' not in test_document or 'data' not in test_document['pdf']:
            return jsonify({
                'status': 'error',
                'message': f'Test with name "{test_name}" not found or invalid PDF data in the database.'
            }), 404
        print("Checkpoint 0")
        # Decode the PDF data from Base64
        # pdf_data = base64.b64decode(test_document['pdf']['data'])
        # pdf_stream = io.BytesIO(pdf_data)
        # raw_pdf_data = pdf_stream.getvalue()
        # print("raw")
# Pass raw bytes to get_pdf_text
        data = test_document['pdf']['data']
        #print(data)
        doc_name = test_name+".pdf"
        with open(doc_name, "wb") as f:
            f.write(data)
        #print(data)
        with open(doc_name, "rb") as f:
            pdf_stream = BytesIO(f.read())
            pdf_reader = PdfReader(pdf_stream)
            raw_text = ""
            for page in pdf_reader.pages:
                raw_text += page.extract_text()
        print(raw_text)
        print("Checkpoint 1")
        if not raw_text.strip():
            return jsonify({
                'status': 'error',
                'message': 'The PDF has no extractable text.'
            }), 400
        
        # Split text into chunks and create vector store
        text_chunks = get_text_chunks(raw_text)
        create_vector_store(text_chunks, test_name)
        
        
        os.remove(doc_name)
        
        return jsonify({
            'status': 'success',
            'message': f'Test "{test_name}" processed successfully.'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'An error occurred: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/evaluate_answer', methods=['POST'])
def evaluate_answer():
    """Endpoint to evaluate a user's answer"""
    try:
        data = request.json
        test_name = data.get('test_name')
        question = data.get('question')
        user_answer = data.get('user_answer')
        print(test_name)
        # Load vector store for the test
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        
        vector_dir = "vector_databases"
        if not os.path.exists(vector_dir):
            return jsonify({
                'status': 'error',
                'message': 'Vector database directory does not exist.'
            }), 400

        # Load vector store for the test
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_name = os.path.join(vector_dir, f"faiss_index_{test_name}")
        
        if not os.path.exists(vector_name):
            return jsonify({
                'status': 'error',
                'message': f'No vector store found for test "{test_name}".'
            }), 404
        
        
        print("Vec:",vector_name)
        vector_store = FAISS.load_local(
            vector_name, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Get relevant documents
        docs = vector_store.similarity_search(question)
        print("C1")
        # Get evaluation chain
        chain = get_conversational_chain()
        print("c8")
        # Get initial evaluation
        response = chain(
            {
                "input_documents": docs,
                "question": question,
                "user_answer": user_answer
            },
            return_only_outputs=True
        )
        print("c2")
        output_text = response["output_text"]
        
        # If context is insufficient, use Gemini directly
        if "Answer is not available in the context." in output_text:
            model = genai.GenerativeModel("gemini-1.5-flash")
            query = f'''
                You are an evaluator responsible for assessing a provided answer for evaluation.
                Your task is to generate an informed evaluation based on the provided question and answer.
                start your response with a rating between 0 - 10 and on next line continue with the evaluation.
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
