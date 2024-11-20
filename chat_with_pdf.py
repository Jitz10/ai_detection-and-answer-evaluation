import streamlit as st
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





load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

mongo_client = os.getenv("mongo_client")
#mongo setup
client = MongoClient(mongo_client)

# Select the database
db = client['fosip']

# Select the collection
#collection = db['my_collection']




def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are an evaluator responsible for assessing a user answer based on the provided question, context, and evaluation rubrics. Your goal is to provide an accurate score and explanation based on the rubrics and the information available.

    Instructions:
    - Carefully read the context to determine if it contains sufficient information to evaluate the user’s answer.
    - If the context lacks adequate information to evaluate the answer, respond with: "Answer is not available in the context." and do not proceed further.
    - If the context is sufficient, assign a score from 0 to 10 at the beginning of your response (formatted as the first two characters). This score should be based on the rubrics and should be followed by a clear, detailed explanation justifying the score. Highlight strengths and weaknesses as they relate to the rubrics.
    - Additionally, consider any provided reference answers along with their scores for comparison. If no references are provided, rely solely on the context and rubrics for your evaluation.

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
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, rubrics, user_answer):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Allow dangerous deserialization
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question, "rubrics": rubrics, "user_answer": user_answer},
        return_only_outputs=True
    )

    output_text = response["output_text"]

    # Check if "Answer is not available in the context." is part of the response
    if "Answer is not available in the context." in output_text:
        # Use the GenerativeModel for an additional API call to Gemini
        model = genai.GenerativeModel("gemini-1.5-flash")
        query = f'''
            You are an evaluator responsible for assessing a provided answer for evaluation. Your task is to generate an informed evaluation based on the provided question, answer, and rubrics.

            Instructions:
            - Review the question and user-provided answer carefully.
            - Use the rubrics to evaluate the answer, ensuring the scoring is clear, accurate, and well-explained.
            - Start your evaluation by assigning a score from 0 to 10 (as the first two characters), followed by an explanation detailing strengths and weaknesses according to the rubrics.
            - If applicable, use external knowledge to ensure a thorough evaluation, given that the original context was insufficient.

            Components:

            Rubrics for Evaluation:
            {rubrics}

            Question:
            {user_question}

            User Answer:
            {user_answer}

            Evaluation:
        '''

        st.write(query)

        genai_response = model.generate_content(query)
        supplemental_answer = genai_response.text
        
        # Append the supplemental answer to the output
        st.write(supplemental_answer)
    else:
    # Display the response in Streamlit
        st.write("Reply: ", output_text)

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")
    rubrics = st.text_area("Enter evaluation rubrics:")
    user_answer = st.text_area("Enter the user answer:")

    if user_question and rubrics and user_answer:
        user_input(user_question, rubrics, user_answer)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()