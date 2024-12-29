import os
import json
import chromadb
import numpy as np
from dotenv import load_dotenv
import gradio as gr
from groq import Groq
import torch
from transformers import AutoTokenizer, AutoModel
import logging

# Load environment variables
load_dotenv()

# List of API keys for Groq
api_keys = [
    os.getenv("GROQ_API_KEY"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
]

if not any(api_keys):
    raise ValueError("At least one GROQ_API_KEY environment variable must be set.")

# Initialize Groq client with the first API key
current_key_index = 0
client = Groq(api_key=api_keys[current_key_index])

# Define Groq-based model with fallback
class GroqChatbot:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.client = Groq(api_key=self.api_keys[self.current_key_index])

    def switch_key(self):
        """Switch to the next API key in the list."""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.client = Groq(api_key=self.api_keys[self.current_key_index])
        print(f"Switched to API key index {self.current_key_index}")

    def get_response(self, prompt):
        """Get a response from the API, switching keys on failure."""
        while True:
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    model="llama3-70b-8192",
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error: {e}")
                self.switch_key()
                if self.current_key_index == 0:
                    return "All API keys have been exhausted. Please try again later."

    def text_to_embedding(self, text):
        """Convert text to embedding using the current model."""
        try:
            # Load the model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-3.2-1B")
            model = AutoModel.from_pretrained("NousResearch/Llama-3.2-1B")

            # Move model to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()

            # Ensure tokenizer has a padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Tokenize the text
            encoded_input = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)

            # Generate embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)
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

                print(f"Generated embedding for text: {text}")
                return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

# Modify LocalEmbeddingStore to use ChromaDB
class LocalEmbeddingStore:
    def __init__(self, storage_dir="./chromadb_storage"):
        self.client = chromadb.PersistentClient(path=storage_dir)  # Use ChromaDB client with persistent storage
        self.collection_name = "chatbot_docs"  # Collection for storing embeddings
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_embedding(self, doc_id, embedding, metadata):
        """Add a document and its embedding to ChromaDB."""
        self.collection.add(
            documents=[doc_id],  # Document ID for identification
            embeddings=[embedding],  # Embedding for the document
            metadatas=[metadata],  # Optional metadata
            ids=[doc_id]  # Same ID as document ID
        )
        print(f"Added embedding for document ID: {doc_id}")

    def search_embedding(self, query_embedding, num_results=3):
        """Search for the most relevant document based on embedding similarity."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=num_results
        )
        print(f"Search results: {results}")
        return results['documents'], results['distances']  # Returning both document IDs and distances

# Modify RAGSystem to integrate ChromaDB search
class RAGSystem:
    def __init__(self, groq_client, embedding_store):
        self.groq_client = groq_client
        self.embedding_store = embedding_store

    def get_most_relevant_document(self, query_embedding):
        """Retrieve the most relevant document based on cosine similarity."""
        docs, distances = self.embedding_store.search_embedding(query_embedding)
        if docs:
            return docs[0], distances[0]  # Return the most relevant document
        return None, None

    def chat_with_rag(self, user_input):
        """Handle the RAG process."""
        query_embedding = self.groq_client.text_to_embedding(user_input)
        if query_embedding is None or query_embedding.size == 0:
            return "Failed to generate embeddings."

        context_document_id, similarity_score = self.get_most_relevant_document(query_embedding)
        if not context_document_id:
            return "No relevant documents found."

        # Assuming metadata retrieval works
        context_metadata = f"Metadata for {context_document_id}"  # Placeholder, implement as needed

        prompt = f"""Context (similarity score {similarity_score:.2f}):
{context_metadata}

User: {user_input}
AI:"""
        return self.groq_client.get_response(prompt)

# Initialize components
embedding_store = LocalEmbeddingStore(storage_dir="./chromadb_storage")
chatbot = GroqChatbot(api_keys=api_keys)
rag_system = RAGSystem(groq_client=chatbot, embedding_store=embedding_store)

# Gradio UI
def chat_ui(user_input, chat_history):
    """Handle chat interactions and update history."""
    if not user_input.strip():
        return chat_history
    ai_response = rag_system.chat_with_rag(user_input)
    chat_history.append((user_input, ai_response))
    return chat_history

# Gradio interface
with gr.Blocks() as demo:
    chat_history = gr.Chatbot(label="Groq Chatbot with RAG", elem_id="chatbox")
    user_input = gr.Textbox(placeholder="Enter your prompt here...")
    submit_button = gr.Button("Submit")
    submit_button.click(chat_ui, inputs=[user_input, chat_history], outputs=chat_history)

if __name__ == "__main__":
    demo.launch()
