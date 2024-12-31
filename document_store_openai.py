import os
import openai
import chromadb
from chromadb.config import Settings
import numpy as np
import logging
import fitz  # PyMuPDF for extracting text from PDFs
from dotenv import load_dotenv  # Import dotenv

# Load environment variables from .env file
load_dotenv()

# Enable logging for ChromaDB and debugging
logging.basicConfig(level=logging.DEBUG)

class DocumentStore:
    def __init__(self, chunk_size=8000, chunk_overlap=200):
        print("Initializing DocumentStore...")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Ensure the directory exists for ChromaDB
        self.chromadb_storage_dir = "./chromadb_storage"
        if not os.path.exists(self.chromadb_storage_dir):
            os.makedirs(self.chromadb_storage_dir)
            print(f"Created directory: {self.chromadb_storage_dir}")

        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(path=self.chromadb_storage_dir)
        try:
            self.collection = self.client.get_or_create_collection("chatbot_docs")
            print("Collection initialized successfully")
        except Exception as e:
            print(f"Error initializing Chroma collection: {str(e)}")
            raise

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY in your .env file.")
        self.openai_client = openai.OpenAI(api_key=api_key)

    def chunk_text(self, text):
        """Split text into overlapping chunks of approximately equal size."""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            # Find the end of the chunk
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to find a good breaking point
            if end < text_len:
                # Look for a period, question mark, or exclamation mark followed by a space
                # within the last 100 characters of the chunk
                look_back = min(100, self.chunk_size)
                last_period = -1
                for i in range(end, max(end - look_back, start), -1):
                    if i < text_len and text[i-1] in '.!?' and (i == text_len or text[i].isspace()):
                        last_period = i
                        break
                
                if last_period != -1:
                    end = last_period

            # Add the chunk
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)

            # Move the start pointer, accounting for overlap
            start = end - self.chunk_overlap

        return chunks

    def text_to_embedding(self, text):
        """Convert text to numerical embedding vector using OpenAI's API."""
        print(f"Converting text to embedding (length: {len(text)} chars)")
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            )
            embedding = response.data[0].embedding
            
            # Ensure embedding dimensionality is 3072
            if len(embedding) != 3072:
                raise ValueError(f"Embedding dimensionality mismatch. Expected 3072, got {len(embedding)}")
            print(f"Generated embedding length: {len(embedding)}")
            return embedding
        except Exception as e:
            print(f"Error in text_to_embedding: {str(e)}")
            raise

    def add_document(self, doc_text, doc_id_prefix):
        """Add a single document to the document store."""
        try:
            print(f"\nProcessing document: {doc_id_prefix}")

            if not doc_text or not isinstance(doc_text, str):
                raise ValueError(f"Invalid document text for {doc_id_prefix}")

            # Split text into chunks
            chunks = self.chunk_text(doc_text)
            print(f"Split document into {len(chunks)} chunks")

            # Process each chunk
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id_prefix}_chunk_{i}"
                
                # Convert chunk to embedding
                embedding = self.text_to_embedding(chunk)

                # Add to ChromaDB
                self.collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    ids=[chunk_id]
                )
                print(f"Added chunk {i+1}/{len(chunks)}")

            print(f"Successfully processed document: {doc_id_prefix}")

        except Exception as e:
            print(f"Error adding document {doc_id_prefix}: {str(e)}")
            raise

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        try:
            print(f"Extracting text from: {pdf_path}")
            doc = fitz.open(pdf_path)
            text = " ".join(page.get_text() for page in doc)
            doc.close()

            if not text.strip():
                print("Warning: Extracted empty text from PDF")
                return None

            print(f"Successfully extracted {len(text)} characters")
            return text

        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return None

    def add_pdfs_from_directory(self, directory_path):
        """Add all PDFs from a directory."""
        print(f"\nProcessing directory: {directory_path}")

        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}")
            return

        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        print(f"Found {len(pdf_files)} PDF files")

        for filename in pdf_files:
            pdf_path = os.path.join(directory_path, filename)
            print(f"\nProcessing: {filename}")

            # Extract text from PDF
            pdf_text = self.extract_text_from_pdf(pdf_path)

            if pdf_text:
                # Create document ID from filename
                doc_id = os.path.splitext(filename)[0]
                self.add_document(pdf_text, doc_id)

    def query_documents(self, query_text, n_results=5):
        """Query the document store and retrieve relevant documents."""
        try:
            print(f"Querying documents for: {query_text}")

            # Convert query text to embedding
            query_embedding = self.text_to_embedding(query_text)

            # Perform the query
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            print(f"Retrieved {len(results['documents'][0])} documents")
            return results['documents'][0]

        except Exception as e:
            print(f"Error querying documents: {str(e)}")
            return []

def main():
    """Main function to demonstrate usage."""
    try:
        # Initialize document store with custom chunk size and overlap
        doc_store = DocumentStore(chunk_size=8000, chunk_overlap=200)

        # Set up the directory path
        directory_path = "./pdfs"

        # Add PDFs from directory
        doc_store.add_pdfs_from_directory(directory_path)

        # Example Query
        query_text = "what is swarms?"
        relevant_docs = doc_store.query_documents(query_text)
        for i, doc in enumerate(relevant_docs):
            print(f"\nDocument {i+1}:\n{doc[:500]}...")  # Print first 500 characters of each document

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
