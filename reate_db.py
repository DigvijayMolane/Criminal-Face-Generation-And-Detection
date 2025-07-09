import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('criminal_detection.db')
cursor = conn.cursor()

# Create the users table
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
''')

# Create the criminals table
cursor.execute('''
CREATE TABLE IF NOT EXISTS criminals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER NOT NULL,
    gender TEXT NOT NULL,
    crime TEXT NOT NULL,
    image_path TEXT NOT NULL,
    facial_embedding BLOB NOT NULL  -- Storing face encodings as binary data
)
''')

# Commit changes and close connection
conn.commit()
conn.close()

print("Database and tables created successfully!")
