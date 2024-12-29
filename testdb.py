import sqlite3
import numpy as np

def test_chromadb_database(database_path="./chromadb_storage/chroma.sqlite3", collection_name="chatbot_docs"):
    # Open a connection to the SQLite database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    print(f"Testing database at: {database_path}\n")

    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables in the database:", [table[0] for table in tables])

    # Print schema for each table
    for table in tables:
        print(f"\nSchema for table '{table[0]}':")
        cursor.execute(f"PRAGMA table_info({table[0]});")
        columns = cursor.fetchall()
        for column in columns:
            print(column)

    # Query data from the 'collections' table
    print("\n--- Collections Table ---")
    cursor.execute("SELECT * FROM collections;")
    collections = cursor.fetchall()
    print("Collections:")
    for collection in collections:
        print(collection)

    # Validate that the target collection exists
    cursor.execute("SELECT id FROM collections WHERE name = ?;", (collection_name,))
    collection_id_row = cursor.fetchone()
    if not collection_id_row:
        print(f"Collection '{collection_name}' not found in the database.")
        conn.close()
        return
    collection_id = collection_id_row[0]
    print(f"Collection ID for '{collection_name}': {collection_id}")

    # Query embeddings linked to the collection
    print("\n--- Embeddings Table ---")
    cursor.execute("SELECT * FROM embeddings WHERE segment_id = ?;", (collection_id,))
    embeddings = cursor.fetchall()
    print(f"Embeddings for collection '{collection_name}':")
    for embedding in embeddings:
        print(embedding)

    # Check metadata for the embeddings
    print("\n--- Embedding Metadata ---")
    cursor.execute("""
        SELECT e.embedding_id, m.key, m.string_value 
        FROM embeddings e 
        LEFT JOIN embedding_metadata m ON e.embedding_id = m.id 
        WHERE e.segment_id = ?;
    """, (collection_id,))
    metadata = cursor.fetchall()
    print(f"Metadata for embeddings in collection '{collection_name}':")
    for meta in metadata:
        print(meta)

    # Perform a test retrieval with a random query embedding
    print("\n--- Test Document Retrieval ---")
    if embeddings:
        # Example query embedding (same size as stored embeddings, e.g., 1024)
        query_embedding = np.random.rand(1024).astype(np.float32)

        def cosine_similarity(a, b):
            """Calculate cosine similarity between two vectors."""
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            return dot_product / (norm_a * norm_b)

        # Convert embeddings from database and find the most similar
        most_similar_id = None
        highest_similarity = -1
        for embedding_row in embeddings:
            embedding_id = embedding_row[2]
            embedding_blob = embedding_row[3]
            embedding_vector = np.frombuffer(embedding_blob, dtype=np.float32)
            similarity = cosine_similarity(query_embedding, embedding_vector)
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_id = embedding_id

        if most_similar_id:
            print(f"Most similar document ID: {most_similar_id} with similarity: {highest_similarity:.4f}")
            # Fetch metadata for the most similar document
            cursor.execute("SELECT key, string_value FROM embedding_metadata WHERE id = ?;", (most_similar_id,))
            most_similar_metadata = cursor.fetchall()
            print("Metadata for the most similar document:")
            for meta in most_similar_metadata:
                print(meta)
        else:
            print("No similar document found.")
    else:
        print(f"No embeddings found for collection '{collection_name}'.")

    # Close the connection
    conn.close()
    print("\nTesting complete.")

# Run the testing suite
test_chromadb_database()
