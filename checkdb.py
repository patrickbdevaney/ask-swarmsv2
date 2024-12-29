import sqlite3

# Open a connection to the SQLite database
conn = sqlite3.connect("./chromadb_storage/chroma.sqlite3")
cursor = conn.cursor()

# Get the list of tables in the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Print the names of tables
print("Tables:", tables)

# For each table, print its schema
for table in tables:
    print(f"\nSchema for table '{table[0]}':")
    cursor.execute(f"PRAGMA table_info({table[0]});")
    columns = cursor.fetchall()
    for column in columns:
        print(column)

# Close the connection
conn.close()
