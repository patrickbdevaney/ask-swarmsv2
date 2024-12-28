import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import logging
import fitz  # PyMuPDF for extracting text from PDFs

# Enable logging for ChromaDB
logging.basicConfig(level=logging.DEBUG)

class DocumentStore:
    def __init__(self):
        print("Initializing DocumentStore...")

        # Ensure the directory exists for ChromaDB (using a relative path to store in the same location)
        self.chromadb_storage_dir = "./chromadb_storage"
        
        # Ensure the directory exists
        if not os.path.exists(self.chromadb_storage_dir):
            os.makedirs(self.chromadb_storage_dir)
            print(f"Created directory: {self.chromadb_storage_dir}")

        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(path=self.chromadb_storage_dir)
        try:
            # Try to get existing collection or create a new one
            self.collection = self.client.get_or_create_collection("chatbot_docs")
            print("Collection initialized successfully")
        except Exception as e:
            print(f"Error initializing Chroma collection: {str(e)}")
            raise

        print("Loading model and tokenizer...")
        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-3.2-1B")
        self.model = AutoModel.from_pretrained("NousResearch/Llama-3.2-1B")
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Ensure tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def text_to_embedding(self, text):
        """Convert text to numerical embedding vector."""
        print(f"Converting text to embedding (first 100 chars): {text[:100]}...")

        try:
            # Tokenize the text
            encoded_input = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)

            print("Text tokenized successfully")

            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                sentence_embeddings = model_output.last_hidden_state
                
                # Mean pooling
                attention_mask = encoded_input['attention_mask']
                mask = attention_mask.unsqueeze(-1).expand(sentence_embeddings.size()).float()
                masked_embeddings = sentence_embeddings * mask
                summed = torch.sum(masked_embeddings, dim=1)
                summed_mask = torch.clamp(torch.sum(attention_mask, dim=1).unsqueeze(-1), min=1e-9)
                mean_pooled = (summed / summed_mask).squeeze()
                
                # Move to CPU and convert to numpy
                embedding = mean_pooled.cpu().numpy()
                
                # Normalize the embedding vector
                embedding = embedding / np.linalg.norm(embedding)

                print(f"Generated embedding shape: {embedding.shape}")
                print(f"Embedding sample (first 5 values): {embedding[:5]}")

                return embedding

        except Exception as e:
            print(f"Error in text_to_embedding: {str(e)}")
            raise

    def add_document(self, doc_text, doc_id):
        """Add a single document to the document store."""
        try:
            print(f"\nProcessing document: {doc_id}")

            if not doc_text or not isinstance(doc_text, str):
                raise ValueError(f"Invalid document text for {doc_id}")

            # Convert text to embedding
            embedding = self.text_to_embedding(doc_text)

            # Verify embedding format
            if not isinstance(embedding, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(embedding)}")

            # Ensure embedding is in the correct format (list of lists)
            embedding_list = [embedding.tolist()]

            print(f"Embedding list type: {type(embedding_list)}")
            print(f"Embedding list shape: {len(embedding_list)}x{len(embedding_list[0])}")

            # Add to ChromaDB
            self.collection.add(
                embeddings=embedding_list,
                documents=[doc_text],
                ids=[doc_id]
            )
            print(f"Successfully added document: {doc_id}")

        except Exception as e:
            print(f"Error adding document {doc_id}: {str(e)}")
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

    def add_pdfs_from_directory(self, directory_path, batch_size=50):
        """Add all PDFs from a directory in batches."""
        print(f"\nProcessing directory: {directory_path}")

        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}")
            return

        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        print(f"Found {len(pdf_files)} PDF files")

        batch_embeddings = []
        batch_texts = []
        batch_ids = []

        for i, filename in enumerate(pdf_files):
            pdf_path = os.path.join(directory_path, filename)
            print(f"\nProcessing: {filename}")

            # Extract text from PDF
            pdf_text = self.extract_text_from_pdf(pdf_path)

            if pdf_text:
                # Create document ID from filename
                doc_id = os.path.splitext(filename)[0]
                batch_embeddings.append(self.text_to_embedding(pdf_text))
                batch_texts.append(pdf_text)
                batch_ids.append(doc_id)

                # If batch size reached, add to collection
                if (i + 1) % batch_size == 0 or i + 1 == len(pdf_files):
                    self.collection.add(
                        embeddings=batch_embeddings,
                        documents=batch_texts,
                        ids=batch_ids
                    )
                    print(f"Batch of {len(batch_embeddings)} documents added to collection.")
                    batch_embeddings.clear()
                    batch_texts.clear()
                    batch_ids.clear()

def main():
    """Main function to demonstrate usage."""
    try:
        # Initialize document store
        doc_store = DocumentStore()

        # Set up the directory path
        directory_path = "./pdfs"

        # Add PDFs from directory
        doc_store.add_pdfs_from_directory(directory_path)

        # Example Query (Optional)
        query_results = doc_store.collection.query(
            query_texts=["Your query text here"],
            n_results=5
        )
        print(query_results)

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
