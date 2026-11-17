import sqlite3

def get_all_messages(db_path="conversation_memory.db"):
    """Fetch all messages from the database."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT role, message, created_at FROM conversation_logs ORDER BY created_at ASC")
    messages = c.fetchall()
    conn.close()
    return messages

# Example usage
#messages = get_all_messages()
#for role, message, created_at in messages:
#    print(f"[{created_at}] {role}: {message}")
