import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Set up a directory for file uploads
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_pdf_text(pdf_file_path):
    """Extracts text from a single PDF file."""
    try:
        pdf_reader = PdfReader(pdf_file_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return str(e)

def split_text_to_chunks(raw_text):
    """Splits text into smaller chunks."""
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(raw_text)
        return chunks
    except Exception as e:
        return str(e)

def create_vectorstore(text_chunks):
    """Creates a FAISS vector store using text chunks."""
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        vectorstore = FAISS.from_texts(text_chunks, embeddings)
        return vectorstore
    except Exception as e:
        return str(e)

def generate_conversation_chain(vectorstore):
    """Creates a conversational retrieval chain."""
    try:
        llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        return str(e)

@app.route("/process", methods=["POST"])
def process_request():
    """API endpoint to process a PDF file and answer a user question."""
    if "pdf" not in request.files or "question" not in request.form:
        return jsonify({"error": "Missing PDF file or question in the request."}), 400

    pdf_file = request.files["pdf"]
    user_question = request.form["question"]

    if pdf_file.filename == "":
        return jsonify({"error": "No file provided."}), 400

    try:
        # Save the uploaded PDF file
        filename = secure_filename(pdf_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        pdf_file.save(file_path)

        # Process the PDF file
        raw_text = extract_pdf_text(file_path)
        if not raw_text:
            return jsonify({"error": "No text found in the uploaded PDF."}), 400

        # Split text into chunks and create vector store
        text_chunks = split_text_to_chunks(raw_text)
        vectorstore = create_vectorstore(text_chunks)
        if isinstance(vectorstore, str):
            return jsonify({"error": vectorstore}), 500

        # Generate conversation chain and get response
        conversation_chain = generate_conversation_chain(vectorstore)
        if isinstance(conversation_chain, str):
            return jsonify({"error": conversation_chain}), 500

        prompt = (
            f"You are an AI assistant. Based on the uploaded document, answer the following question: "
            f"'{user_question}'"
        )
        response = conversation_chain.invoke({"question": prompt})

        if "chat_history" in response:
            return jsonify({
                "response": response["chat_history"][-1]["content"]
            })
        else:
            return jsonify({"error": "No response from the model."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
