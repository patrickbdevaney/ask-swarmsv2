import os
import openai
import chromadb
import numpy as np
from dotenv import load_dotenv
import gradio as gr
import logging

# Load environment variables
load_dotenv()

# OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable must be set.")

# Define OpenAI-based model
class OpenAIChatbot:
    def __init__(self):
        self.embedding_model = "text-embedding-3-large"  # OpenAI model with 3072 dimensions
        self.chat_model = "gpt-4o"
        self.api_key = api_key

    def get_response(self, prompt):
        """Get a response from OpenAI GPT-4 model."""
        try:
            openai.api_key = self.api_key
            response = openai.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            # Correctly access the message content in the response
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error: Unable to generate a response."

    def text_to_embedding(self, text):
        """Convert text to embedding using OpenAI embedding model."""
        try:
            openai.api_key = self.api_key
            response = openai.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            # Access the embedding using the 'data' attribute
            embedding = np.array(response.data[0].embedding)
            print(f"Generated embedding for text: {text}")
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

# Modify LocalEmbeddingStore to ensure correct dimensionality (3072) in ChromaDB
class LocalEmbeddingStore:
    def __init__(self, storage_dir="./chromadb_storage_openai_upgrade"):
        # Use ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(path=storage_dir)
        self.collection_name = "chatbot_docs"

        # Get the collection without adding new embeddings
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def search_embedding(self, query_embedding, num_results=3):
        """Search for the most relevant document based on embedding similarity."""
        if query_embedding.shape[0] != 3072:
            raise ValueError("Query embedding dimensionality must be 3072.")

        print(f"Query embedding: {query_embedding}")  # Debugging: Log the query embedding
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],  # Ensure embeddings are converted to list format
            n_results=num_results
        )
        print(f"Search results: {results}")  # Debugging: Print results to check for any issues
        return results['documents'], results['distances']

# Modify RAGSystem to integrate ChromaDB search
class RAGSystem:
    def __init__(self, openai_client, embedding_store):
        self.openai_client = openai_client
        self.embedding_store = embedding_store

    def get_most_relevant_document(self, query_embedding, similarity_threshold=0.7):
        """Retrieve the most relevant document based on cosine similarity."""
        docs, distances = self.embedding_store.search_embedding(query_embedding)
        # Check if the results are empty or have low relevance
        if not docs or not distances or distances[0][0] < similarity_threshold:
            print("No relevant documents found or similarity is too low.")
            return None, None  # Return None if no relevant documents found
        return docs[0], distances[0][0]  # Return the most relevant document and the first distance value

    def chat_with_rag(self, user_input):
        """Handle the RAG process."""
        query_embedding = self.openai_client.text_to_embedding(user_input)
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
        return self.openai_client.get_response(prompt)

# Initialize components
embedding_store = LocalEmbeddingStore(storage_dir="./chromadb_storage_openai_upgrade")
chatbot = OpenAIChatbot()
rag_system = RAGSystem(openai_client=chatbot, embedding_store=embedding_store)

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
    chat_history = gr.Chatbot(label="OpenAI Chatbot with RAG", elem_id="chatbox")
    user_input = gr.Textbox(placeholder="Enter your prompt here...")
    submit_button = gr.Button("Submit")
    submit_button.click(chat_ui, inputs=[user_input, chat_history], outputs=chat_history)

if __name__ == "__main__":
    demo.launch()
