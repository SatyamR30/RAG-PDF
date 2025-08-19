import sqlite3
from app.config import settings

def init_db():
    conn = sqlite3.connect(settings.DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    filename TEXT,
                    filepath TEXT
                )""")
    conn.commit()
    conn.close()

def insert_metadata(filename, filepath):
    conn = sqlite3.connect(settings.DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO documents (filename, filepath) VALUES (?, ?)", (filename, filepath))
    conn.commit()
    conn.close()

def get_metadata():
    conn = sqlite3.connect(settings.DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM documents")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "filename": r[1], "filepath": r[2]} for r in rows]
