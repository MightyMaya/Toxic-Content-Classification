import sqlite3


def initialize_db():
    conn = sqlite3.connect("entries.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS entries \
                   (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, category TEXT)")
    conn.commit()
    conn.close()

def insert(entry):

    text = entry["text"]
    category = entry["category"]

    conn = sqlite3.connect("entries.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO entries (text,category) VALUES (?,?)", (text, category))

    conn.commit()
    conn.close()

# function that returns all database entries
def get_entries():
    conn = sqlite3.connect("entries.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id,text,category FROM entries")

    rows = cursor.fetchall()
    conn.close()
    return rows