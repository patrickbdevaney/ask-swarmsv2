# ASK-SWARMS: How to Convert Files to PDFs, Store Embeddings in ChromaDB, and Use a Groq-powered Chatbot with RAG

This repository contains scripts and tools that allow you to convert text files into PDF files, store those PDFs as embeddings in ChromaDB, and create a Groq-powered chatbot using a Retrieval-Augmented Generation (RAG) system for querying relevant documents.

## Overview

This project is broken down into three main parts:

1. **filetopdf.py**: A script that converts all text files in a given directory to PDF files.
2. **document_store.py**: A script that extracts text from PDFs, generates embeddings using the Llama model, and stores them in ChromaDB.
3. **ask_swarms.py**: A chatbot that uses Groq and ChromaDB to answer queries based on documents embedded in the database.

## Setup

### Prerequisites

Ensure you have the following Python libraries installed:

- `chromadb`
- `torch`
- `transformers`
- `fpdf`
- `fitz` (PyMuPDF)
- `gradio`
- `dotenv`
- `groq`

You can install the dependencies using `pip`:

```bash
pip install chromadb torch transformers fpdf PyMuPDF gradio dotenv groq
```

Additionally, you will need an API key for Groq. Set up your `.env` file with the following variables:

```dotenv
GROQ_API_KEY=your_api_key_here
```

### Directory Structure

Ensure that your directory structure follows this format:

```
project_root/
│
├── files/                # Source directory with text files to convert
│
├── pdfs/                 # Output directory for generated PDFs
│
├── chromadb_storage/     # Directory where ChromaDB will store embeddings
│
├── .env                  # Your Groq API keys
├── filetopdf.py          # Script to convert text files to PDFs
├── document_store.py     # Script to store PDFs as embeddings in ChromaDB
└── ask_swarms.py         # Script for the Groq-powered chatbot
```

## Step 1: Convert Text Files to PDFs

The `filetopdf.py` script converts all text files in the `files/` directory to PDF format.

### Usage:

```bash
python filetopdf.py
```

This will scan the `files/` directory, convert each text file to a PDF, and save the PDFs to the `pdfs/` directory. The output PDF filenames will match the input text file names, with the `.pdf` extension.

### Code Breakdown:

- `text_to_pdf()`: Converts the provided text into a PDF file.
- `convert_files_to_pdfs()`: Scans the `source_dir`, converts the text files to PDFs, and saves them in `output_dir`.

## Step 2: Store PDFs as Embeddings in ChromaDB

The `document_store.py` script extracts text from the PDFs, generates embeddings using the Llama model, and stores them in ChromaDB.

### Usage:

```bash
python document_store.py
```

This will:

1. Extract text from each PDF in the `pdfs/` directory.
2. Generate numerical embeddings for each document using the Llama model.
3. Store these embeddings in ChromaDB for later retrieval.

### Code Breakdown:

- `extract_text_from_pdf()`: Extracts the text from the provided PDF file using `fitz` (PyMuPDF).
- `text_to_embedding()`: Converts the extracted text into numerical embeddings using the Llama model.
- `add_document()`: Adds the document and its embeddings to ChromaDB.
- `add_pdfs_from_directory()`: Processes all PDF files in a given directory, generating embeddings for each.

## Step 3: Querying with a Groq-powered Chatbot Using RAG

The `ask_swarms.py` script implements a chatbot that uses a Retrieval-Augmented Generation (RAG) approach. It retrieves the most relevant document based on a query and generates a response using the Groq API.

### Usage:

```bash
python ask_swarms.py
```

This will launch a Gradio interface where you can interact with the chatbot. It will search for the most relevant document from the ChromaDB collection and generate a response based on the context of that document.

### Code Breakdown:

- `GroqChatbot`: Uses the Groq API to generate responses and manage API key rotation.
- `LocalEmbeddingStore`: Manages the ChromaDB client and stores document embeddings.
- `RAGSystem`: Retrieves relevant documents from ChromaDB and combines them with a Groq-generated response for enhanced chatbot functionality.
- `chat_ui()`: Handles user input and updates the chat history in the Gradio UI.

### Gradio Interface:

The Gradio interface consists of:

- A textbox for entering user queries.
- A chat history display showing previous interactions.
- A submit button to send the query.

## Example Workflow

1. **Convert Text Files to PDFs**: Place your text files in the `files/` directory and run `filetopdf.py` to convert them into PDFs stored in `pdfs/`.
   
2. **Store PDFs in ChromaDB**: Run `document_store.py` to extract text from the PDFs, generate embeddings, and store them in ChromaDB.

3. **Interact with the Chatbot**: Run `ask_swarms.py` to launch the chatbot interface. Enter a query, and the chatbot will retrieve the most relevant document from ChromaDB and generate a response.

## Conclusion

This setup allows you to efficiently convert text files to PDFs, store them as embeddings in ChromaDB, and create a Groq-powered chatbot that leverages document retrieval for context-aware responses. This approach is ideal for building a chatbot that can respond with information from a specific document store, such as a set of manuals, guides, or documentation.

You can also download the embeddings generated by document_store.py on the pdfs and put them in the repo: https://drive.google.com/drive/folders/1uyRTLg9thLKFJKzunXIoGtD9B0poI-2l
