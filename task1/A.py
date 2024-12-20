import pdfplumber
import pytesseract
import os
from pdf2image import convert_from_path
from flask import Flask, render_template, request
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
import re

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

UPLOAD_FOLDER = 'D:\\sitafal\\task1\\uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

global vectorstore
vectorstore = None

global current_filename
current_filename = None

def extract_text_from_pdf(file_path):
    page_texts = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    page_texts.append({"text": text, "page": page_num + 1})
                else:
                    try:
                        images = convert_from_path(file_path, first_page=page_num + 1, last_page=page_num + 1)
                        for image in images:
                            ocr_text = pytesseract.image_to_string(image)
                            page_texts.append({"text": ocr_text, "page": page_num + 1})
                    except Exception as ocr_err:
                        print(f"OCR Error on page {page_num + 1}: {ocr_err}")
    except Exception as e:
        print(f"PDF Error: {e}")
        return None
    return page_texts

def create_vectorstore(page_texts):
    global vectorstore
    global page_number_map

    page_number_map = {}
    all_text = ""

    for page_data in page_texts:
        all_text += page_data["text"] + "\n"

    if not all_text:
        return

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.create_documents([all_text])
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    start_index = 0
    for i, doc in enumerate(docs):
        end_index = start_index + len(doc.page_content)
        page_number_map[(start_index, end_index)] = page_texts[0]["page"] 
        start_index = end_index + 1

def get_relevant_sentences(query, k=4):
    global vectorstore
    global page_number_map
    if vectorstore is None:
        return "No PDF uploaded yet."

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    similar_docs = vectorstore.similarity_search(query, k=k)

    relevant_sentences = []
    for doc in similar_docs:
        text = doc.page_content
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        for sentence in sentences:
            if re.search(r'\b' + re.escape(query.lower()) + r'\b', sentence.lower()):
                for (start, end), page_num in page_number_map.items():
                    if start <= text.find(sentence) < end:
                        relevant_sentences.append({"sentence": sentence.strip(), "page": page_num})
                        break
    return relevant_sentences

def get_similar_docs(query, k=4):
    global vectorstore
    if vectorstore is None:
        return "No PDF uploaded yet."

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    similar_docs = vectorstore.similarity_search(query, k=k)
    return similar_docs

@app.route('/', methods=['GET', 'POST'])
def index():
    global current_filename
    if request.method == 'POST':
        if 'pdf' not in request.files:
            return "No file part"
        file = request.files['pdf']
        if file.filename == '':
            return "No selected file"
        if file and file.filename.endswith('.pdf'):
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            extracted_text = extract_text_from_pdf(file_path)
            if extracted_text:
                create_vectorstore(extracted_text)
                current_filename = file.filename
                return render_template('index.html', file_uploaded=True, filename=current_filename)
            else:
                return render_template('index.html', file_uploaded=False, error="Error extracting text from PDF.")
        else:
            return render_template('index.html', file_uploaded=False, error="Invalid file format. Please upload a PDF file.")
    return render_template('index.html', file_uploaded=False)

@app.route('/query', methods=['POST'])
def query_pdf():
    global current_filename
    if request.method == 'POST':
        query = request.form.get('query')
        if not query:
            return "No query provided"

        relevant_sentences = get_relevant_sentences(query)
        if isinstance(relevant_sentences, str):
            return render_template('index.html', file_uploaded=True, query=query, results=[{"sentence": relevant_sentences, "page": None}], filename=current_filename)

        return render_template('index.html', file_uploaded=True, query=query, results=relevant_sentences, filename=current_filename)

    return render_template('index.html', file_uploaded=True, filename=current_filename)

if __name__ == '__main__':
    app.run(debug=False,port=8080)