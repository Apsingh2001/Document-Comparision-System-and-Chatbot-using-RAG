# 📄 Document Comparison and Chatbot 🤖 - RAG

This Streamlit application allows users to upload and compare two documents, highlighting differences and providing an AI-powered chatbot to answer questions based on the document contents.

## 🚀 Features
- 📂 **Supports TXT, PDF, DOCX** file formats.
- 🔍 **Document Comparison** with highlighted differences:
  - **Replaced Text**: 🟡➡️🟢
  - **Added Text**: 🔵
  - **Deleted Text**: 🔴
- 🤖 **AI Chatbot (RAG-based)** using FAISS and Ollama.
- 🏗 **Built with:**
  - `Streamlit` for UI.
  - `FAISS` for vector storage.
  - `HuggingFace Embeddings` for text processing.
  - `Ollama LLM` for AI-powered responses.
  - `pdf2image` and `pytesseract` for OCR-based text extraction.

## 📦 Installation
### 1️⃣ Clone the Repository
```sh
$ git clone https://github.com/yourusername/document-comparison-chatbot.git
$ cd document-comparison-chatbot
```
### 2️⃣ Install Dependencies
```sh
$ pip install -r requirements.txt
```
### 3️⃣ Run the Application
```sh
$ streamlit run app.py
```

## 📜 How It Works
### 🔹 Upload Documents
- Upload **two** documents in `.txt`, `.pdf`, or `.docx` format.
- The app extracts text using `fitz` for PDFs, `docx` for Word files, and OCR for image-based PDFs.

### 🔹 Compare Documents
- Click **Compare Documents 📑** to view:
  - **Replaced words** (highlighted in yellow and green).
  - **Added words** (highlighted in blue).
  - **Deleted words** (highlighted in red).
- Download the comparison result as a `.txt` file.

### 🔹 Chat with Documents
- Enter a question in the chatbot field and click **Ask Chatbot 💬**.
- The AI will retrieve relevant document passages and generate an answer.

## 🛠 Configuration
### Ollama LLM Server
Ensure Ollama LLM is running locally:
```sh
$ ollama run llama3.2
```
Change the model in `OLLAMA_BASE_URL` if needed.

### OCR for Image-based PDFs
Ensure `pytesseract` and `pdf2image` are installed:
```sh
$ sudo apt install tesseract-ocr
$ pip install pytesseract pdf2image
```

## 📌 Dependencies
- `streamlit`
- `pytesseract`
- `pdf2image`
- `fitz` (PyMuPDF)
- `FAISS`
- `HuggingFace Embeddings`
- `LangChain`
- `OllamaLLM`

## 🎯 Future Enhancements
- 🔹 Multi-document comparison.
- 🔹 Improved UI with side-by-side view.
- 🔹 Support for Markdown and HTML.

## 📝 License
This project is licensed under the MIT License.

---
👨‍💻 Developed by Abhishek Pramod Singh | 🌟 Star this repo if you found it useful!
