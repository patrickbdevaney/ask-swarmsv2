import os
import json
from dotenv import load_dotenv
import gradio as gr
from groq import Groq
import chromadb
import numpy as np
from langchain.chat_models import ChatOpenAI

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

# Define the RAGSystem class as in the first code snippet
class RAGSystem:
    def __init__(self, groq_client, doc_store):
        self.groq_client = groq_client
        self.doc_store = doc_store

    def get_most_relevant_document(self, query_embedding):
        """Retrieve the most relevant document based on cosine similarity using the document store."""
        # Query ChromaDB collection for most relevant documents
        results = self.doc_store.query(query_embedding, n_results=1)
        if results:
            return results['documents'][0]
        return None

    def chat_with_rag(self, user_input):
        """Main function to handle the RAG process."""
        # Query the embeddings from the Llama model or any other embedding model (already created)
        query_embedding = self.doc_store.text_to_embedding(user_input)
        if not query_embedding:
            return "Failed to generate embeddings."
        
        context = self.get_most_relevant_document(query_embedding)
        if not context:
            return "No relevant documents found."

        prompt = f"Context:\n{context}\n\nUser: {user_input}\nAI:"
        return self.groq_client.get_response(prompt)

# Initialize Chroma client and load the existing collection
client = chromadb.Client()
collection = client.get_collection("chatbot_docs")  # Load the existing collection from ChromaDB

# Initialize Groq chatbot
chatbot = GroqChatbot(api_keys=api_keys)

# Initialize RAG system
rag_system = RAGSystem(chatbot, collection)

# Function to handle RAG process and generate chatbot response
def chat_ui(user_input, chat_history):
    global conversation_history

    # Get response from the RAG system
    ai_response = rag_system.chat_with_rag(user_input)

    # Update conversation history
    chat_history.append(("User", user_input))
    chat_history.append(("AI", ai_response))
    conversation_history.append({"user": user_input, "ai": ai_response})

    return chat_history

# Conversation history
conversation_history = []

# Define Gradio interface functions
def get_conversation_download():
    return json.dumps(conversation_history, indent=4)

# Gradio Layout
with gr.Blocks() as demo:
    with gr.Row():
        chat_history = gr.Chatbot(label="Groq Chatbot with RAG", elem_id="chatbox", height=600)
    with gr.Row():
        user_input = gr.Textbox(placeholder="Enter your prompt here...", lines=6, max_lines=10, elem_id="input-box")
    with gr.Row():
        submit_button = gr.Button("Submit")
        download_button = gr.File(label="Download Conversation", file_name="conversation_history.json")

    submit_button.click(chat_ui, inputs=[user_input, chat_history], outputs=chat_history)
    download_button.click(get_conversation_download, outputs=download_button)

    # Trigger submit when Enter key is pressed in the input field
    user_input.submit(chat_ui, inputs=[user_input, chat_history], outputs=chat_history)

# Launch the Gradio app
demo.launch()
