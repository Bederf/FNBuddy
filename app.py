import os
import pdfplumber
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from annoy import AnnoyIndex
import numpy as np

# Load environment variables from the .env file
load_dotenv()

# Configure Flask app and upload settings
app = Flask(__name__)
UPLOAD_FOLDER = 'info'  # Directory to save uploaded files
ALLOWED_EXTENSIONS = {'txt', 'pdf'}  # Allowed file types
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set up OpenRouter client with the API base and key
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def chunk_text(text, chunk_size=2000):
    """Splits text into chunks to prevent token overflow."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

def load_documents():
    """Loads documents from the info folder."""
    documents = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    info_dir = os.path.join(base_dir, UPLOAD_FOLDER)

    if os.path.exists(info_dir) and os.path.isdir(info_dir):
        for filename in os.listdir(info_dir):
            file_path = os.path.join(info_dir, filename)
            if filename.lower().endswith('.txt'):
                loader = TextLoader(file_path)
                documents += loader.load()

            elif filename.lower().endswith('.pdf'):
                try:
                    with pdfplumber.open(file_path) as pdf:
                        pdf_text = ""
                        for page in pdf.pages:
                            pdf_text += page.extract_text()

                        if pdf_text:
                            for chunk in chunk_text(pdf_text, chunk_size=1000):
                                documents.append(Document(page_content=chunk, metadata={"source": filename}))
                except Exception as e:
                    print(f"Failed to load PDF file {filename}: {e}")
    else:
        print(f"Info directory not found: {info_dir}")

    return documents

documents = load_documents()

# Initialize Annoy index with vector dimension (adjust based on your embedding size)
vector_dim = 512  # Replace with the actual dimension of your embeddings
index = AnnoyIndex(vector_dim, 'angular')  # 'angular' is commonly used; adjust based on your needs

# Placeholder function for embedding; replace this with actual embedding logic
def get_embedding(text):
    # Replace this function with actual embedding generation, e.g., using OpenAIEmbeddings
    return np.random.rand(vector_dim).tolist()

# Add documents to Annoy index
for i, doc in enumerate(documents):
    embedding = get_embedding(doc.page_content)
    index.add_item(i, embedding)
index.build(10)  # You can adjust the number of trees for accuracy/speed tradeoff

def query_openai(prompt):
    """Query OpenAI with the provided prompt."""
    try:
        # Find the top 3 most similar items in the Annoy index
        query_embedding = get_embedding(prompt)
        nearest_indices = index.get_nns_by_vector(query_embedding, 3, include_distances=False)
        nearest_docs = [documents[i] for i in nearest_indices]

        if not nearest_docs:
            return "No relevant documents found.", []

        context = "\n\n".join([doc.page_content for doc in nearest_docs])
        completion = client.chat.completions.create(
            model="mattshumer/reflection-70b:free",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}\nQuestion: {prompt}"}
            ],
            temperature=0.3
        )
        answer = completion.choices[0].message.content

        related_queries_completion = client.chat.completions.create(
            model="mattshumer/reflection-70b:free",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Based on the query '{prompt}', generate 5 related questions."}
            ],
            temperature=0.7
        )

        related_queries = related_queries_completion.choices[0].message.content.strip().split('\n')
        return answer, related_queries

    except Exception as e:
        return f"Error: {e}", []

@app.route('/')
def index():
    return render_template('ai_search.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads and reload documents."""
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Reload documents after uploading a new file
        global documents, index
        documents = load_documents()
        index = AnnoyIndex(vector_dim, 'angular')
        for i, doc in enumerate(documents):
            embedding = get_embedding(doc.page_content)
            index.add_item(i, embedding)
        index.build(10)
        return redirect(url_for('index'))
    return redirect(url_for('index'))

@app.route('/ai_search', methods=['POST'])
def ai_search():
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "Please provide a search query."}), 400

    ai_answer, related_queries = query_openai(query)

    return jsonify({
        "answer": ai_answer,
        "related_queries": related_queries or []
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

