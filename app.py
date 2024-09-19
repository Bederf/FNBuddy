import os
import io
import time
import pickle
from datetime import datetime
from collections import defaultdict
from functools import wraps
import pdfplumber
from quart import Quart, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import asyncio
import aiofiles
import logging
import tracemalloc

# Start tracemalloc for better error tracking
tracemalloc.start()

# Load environment variables from the .env file
load_dotenv()

# Configure Quart app and upload settings
app = Quart(__name__)
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'info')
CACHE_FILE = os.getenv('CACHE_FILE', 'document_cache.pkl')
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set up OpenRouter client with the API base and key
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Global variables
documents = []
vectorstore = None
file_last_modified = {}

def allowed_file(filename: str) -> bool:
    """
    Check if the uploaded file has an allowed extension.

    Args:
        filename (str): Name of the file to be checked.

    Returns:
        bool: True if the file extension is allowed, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def chunk_text(text: str, chunk_size: int = 2000):
    """
    Chunk the text into smaller parts.

    Args:
        text (str): The text to be chunked.
        chunk_size (int): The size of each chunk.

    Yields:
        str: Chunked text.
    """
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

async def process_file(file_path: str, filename: str):
    """
    Process a file to extract and load its content.

    Args:
        file_path (str): Path to the file.
        filename (str): Name of the file.

    Returns:
        list: A list of Document objects extracted from the file.
    """
    if filename.lower().endswith('.txt'):
        loader = TextLoader(file_path)
        return await asyncio.to_thread(loader.load)
    elif filename.lower().endswith('.pdf'):
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                pdf_content = await f.read()
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                pdf_text = ""
                for page in pdf.pages:
                    pdf_text += page.extract_text() or ""
                return [Document(page_content=chunk, metadata={"source": filename}) 
                        for chunk in chunk_text(pdf_text, chunk_size=1000)]
        except Exception as e:
            app.logger.error(f"Failed to load PDF file {filename}: {e}")
            return []

async def load_documents():
    """
    Load documents into the application, either from cache or by processing files.

    If a cached version exists, it loads from cache, otherwise, it processes files
    and updates the cache.
    """
    global documents, vectorstore, file_last_modified
    base_dir = os.path.dirname(os.path.abspath(__file__))
    info_dir = os.path.join(base_dir, UPLOAD_FOLDER)

    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'rb') as f:
                cache_data = pickle.load(f)
            documents = cache_data.get('documents', [])
            file_last_modified = cache_data.get('file_last_modified', {})
            if 'vectorstore_dict' in cache_data and os.path.exists("faiss_index"):
                try:
                    vectorstore = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
                    app.logger.info("Vectorstore loaded successfully.")
                except ValueError as e:
                    app.logger.warning(f"Failed to load vectorstore: {str(e)}. Rebuilding vectorstore.")
                    vectorstore = None
            app.logger.info("Cache loaded successfully.")
        else:
            app.logger.info("No cache file found. Starting from scratch.")
    except (EOFError, pickle.UnpicklingError, KeyError) as e:
        app.logger.warning(f"Error loading cache: {str(e)}. Rebuilding cache.")
        documents = []
        file_last_modified = {}

    if os.path.exists(info_dir) and os.path.isdir(info_dir):
        for filename in os.listdir(info_dir):
            if allowed_file(filename):
                file_path = os.path.join(info_dir, filename)
                last_modified = os.path.getmtime(file_path)
                if filename not in file_last_modified or last_modified > file_last_modified[filename]:
                    app.logger.info(f"Processing file: {filename}")
                    new_docs = await process_file(file_path, filename)
                    documents.extend(new_docs)
                    file_last_modified[filename] = last_modified

        if documents and (vectorstore is None or len(documents) != len(vectorstore.docstore._dict)):
            vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
            vectorstore.save_local("faiss_index")
            
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump({
                    'documents': documents,
                    'file_last_modified': file_last_modified,
                    'vectorstore_dict': 'saved_to_disk'
                }, f)
            app.logger.info("Cache updated and saved.")
    else:
        app.logger.warning(f"Info directory not found: {info_dir}")

    if not vectorstore:
        app.logger.warning("No documents found or loaded. Creating empty vectorstore.")
        vectorstore = FAISS.from_documents([], OpenAIEmbeddings())

    app.logger.info("Document loading completed.")

async def query_openai(prompt: str):
    """
    Query OpenAI with the provided prompt and generate related questions.

    Args:
        prompt (str): The prompt to query OpenAI.

    Returns:
        tuple: The AI-generated answer and related questions.
    """
    try:
        embeddings = OpenAIEmbeddings()
        query_embedding = await asyncio.to_thread(embeddings.embed_query, prompt)

        docs = vectorstore.similarity_search_by_vector(query_embedding, k=3)
        if not docs:
            return "No relevant documents found.", []

        context = "\n\n".join([doc.page_content for doc in docs])
        
        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model="mattshumer/reflection-70b:free",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for FNB Bank employees."},
                {"role": "user", "content": f"Context: {context}\nQuestion: {prompt}"}
            ],
            temperature=0.3
        )
        answer = completion.choices[0].message.content

        # Generate two related questions using a simplified message format
        related_queries_completion = await asyncio.to_thread(
            client.chat.completions.create,
            model="mattshumer/reflection-70b:free",
            messages=[
                {"role": "system", "content": "Generate related questions specific to FNB, dont justify your decision"},
                {"role": "user", "content": f"Given the prompt '{prompt}' and the answer '{answer}', suggest 2 follow-up questions relevant to FNB's context."}
            ],
            temperature=0.5
        )

        # Extract only the related queries, filtering out any unnecessary text
        related_queries = [query.strip() for query in related_queries_completion.choices[0].message.content.split('\n') if query.strip() and not query.lower().startswith('this question')]
        
        return answer, related_queries

    except Exception as e:
        app.logger.error(f"Error in query_openai: {str(e)}", exc_info=True)
        return "An error occurred while processing your request. Please try again later.", []

@app.before_serving
async def startup():
    """
    Runs before the server starts, loading all required documents.
    """
    app.logger.info("Starting document loading...")
    await load_documents()

@app.route('/')
async def index():
    """
    Render the main search page.
    """
    return await render_template('ai_search.html')

@app.route('/upload', methods=['POST'])
async def upload_file():
    """
    Handle file uploads and reload documents after new files are uploaded.
    """
    if 'file' not in (await request.files):
        return redirect(request.url)
    
    file = (await request.files)['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        await file.save(file_path)
        
        # Reload documents after uploading a new file
        await load_documents()
        return redirect(url_for('index'))
    
    return redirect(request.url)

@app.route('/ai_search', methods=['POST'])
async def ai_search():
    """
    Handle AI-powered searches with the provided query.
    """
    data = await request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "Please provide a search query."}), 400

    ai_answer, related_queries = await query_openai(query)

    return jsonify({
        "answer": ai_answer,
        "related_queries": related_queries or []
    })

@app.route('/refresh_documents', methods=['POST'])
async def refresh_documents():
    """
    Manually trigger document refresh.
    """
    app.logger.info("Manually refreshing documents...")
    await load_documents()
    return jsonify({"message": "Documents refreshed successfully"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)
