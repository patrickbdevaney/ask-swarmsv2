import chromadb

class EmbeddingInspector:
    def __init__(self, storage_dir="./chromadb_storage_openai_upgrade", output_file="output.txt"):
        """Initialize the inspector with the ChromaDB client and an output file."""
        self.client = chromadb.PersistentClient(path=storage_dir)
        self.collection_name = "chatbot_docs"  # Modify if you use a different collection name
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.output_file = output_file  # Output file for console logs

    def _log_to_file(self, message):
        """Write a message to the output file."""
        with open(self.output_file, "a") as file:
            file.write(message + "\n")

    def check_embeddings(self):
        """Check the embeddings in the ChromaDB collection and print both documents and embeddings."""
        # Retrieve all documents from the collection
        results = self.collection.get()
        
        # Print the structure of the results to understand the data format
        results_message = f"Results from collection get(): {results}"
        print(results_message)
        self._log_to_file(results_message)
        
        if not results['documents']:
            no_docs_message = "No documents found in the collection."
            print(no_docs_message)
            self._log_to_file(no_docs_message)
            return
        
        # Initialize lists to store document details and embeddings
        documents_and_embeddings = []
        dimensionalities = []  # List to store the dimensionality of each embedding
        
        # Print information about each document and its embedding
        for i, doc in enumerate(results['documents']):
            doc_message = f"Document {i}: {doc}"  # Print the raw document to inspect its structure
            print(doc_message)
            self._log_to_file(doc_message)
            
            # Access the embedding directly if available
            if isinstance(doc, dict):
                embedding = doc.get('embedding')  # This assumes embedding is stored under 'embedding'
                if embedding:
                    embedding_message = f"Document ID: {doc['id']}\nMetadata: {doc.get('metadata', 'No metadata')}\nEmbedding dimensionality: {len(embedding)}\nFirst few values of the embedding: {embedding[:5]}..."
                    print(embedding_message)
                    self._log_to_file(embedding_message)
                    # Store the document and its corresponding embedding
                    documents_and_embeddings.append({
                        'document': doc,
                        'embedding': embedding,
                        'dimensionality': len(embedding)
                    })
                    dimensionalities.append(len(embedding))  # Append the dimensionality to the list
                else:
                    no_embedding_message = "No embedding found for this document."
                    print(no_embedding_message)
                    self._log_to_file(no_embedding_message)
            else:
                unexpected_format_message = f"Unexpected document format: {doc}"
                print(unexpected_format_message)
                self._log_to_file(unexpected_format_message)
            
            print("-" * 50)
            self._log_to_file("-" * 50)

        # Check for dimensionality consistency and log at the bottom of the output
        if dimensionalities:
            dimensionalities_message = f"Dimensionalities of all embeddings: {dimensionalities}"
            print(dimensionalities_message)
            self._log_to_file(dimensionalities_message)

        # Optionally, print or return the collected documents and embeddings
        return documents_and_embeddings

    def get_collection_info(self):
        """Get information about the collection (e.g., number of documents, metadata)."""
        collection_info = self.collection.get()
        num_documents = len(collection_info['documents'])
        collection_info_message = f"Collection '{self.collection_name}' contains {num_documents} documents."
        print(collection_info_message)
        self._log_to_file(collection_info_message)

if __name__ == "__main__":
    # Initialize the inspector
    inspector = EmbeddingInspector(storage_dir="./chromadb_storage_openai_upgrade", output_file="output.txt")
    
    # Check the embeddings and collection info
    inspector.get_collection_info()
    
    # Get both documents and embeddings to compare
    documents_and_embeddings = inspector.check_embeddings()
    
    # Optionally, print or process the documents and embeddings
    if documents_and_embeddings:
        documents_and_embeddings_message = "Documents and Embeddings:\n"
        print(documents_and_embeddings_message)
        inspector._log_to_file(documents_and_embeddings_message)
        
        for item in documents_and_embeddings:
            document_message = f"Document: {item['document']}\nEmbedding Dimensionality: {item['dimensionality']}\n{'-' * 50}"
            print(document_message)
            inspector._log_to_file(document_message)
