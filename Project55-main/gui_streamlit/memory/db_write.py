import threading
import queue
import sqlite3
import time
import requests
import json

# 1) Create a queue that holds messages (or any DB job) to store
db_queue = queue.Queue()

def init_db(db_path="conversation_memory.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversation_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def db_worker(db_path="conversation_memory.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    c = conn.cursor()
    
    while True:
        item = db_queue.get()
        if item is None:
            print("Breaking db_worker")
            break
        try:
            job_type = item["job_type"]
            print(f"Processing job: {job_type}")
            
            if job_type == "store_message":
                role, content = item["role"], item["content"]
                print(f"Storing message: role={role}, content={content}")
                c.execute("INSERT INTO conversation_logs (role, message) VALUES (?, ?)",
                          (role, content))
                conn.commit()
            
            elif job_type == "summarize":
                # Kick off summarization in this background thread
                conversation_text = item["conversation_text"]
                print(f"Summarizing conversation text: {conversation_text}")
                summary = call_ollama_to_summarize(conversation_text)
                print(f"Storing summary: {summary}")
                c.execute("INSERT INTO conversation_memory (summary) VALUES (?)", (summary,))
                conn.commit()
            
        except Exception as e:
            print("DB error:", e)
        finally:
            db_queue.task_done()

    conn.close()


# 2) Start the worker thread once at startup
init_db()
db_thread = threading.Thread(target=db_worker, daemon=True)
db_thread.start()

def store_message_in_db_async(role: str, content: str):
    """
    Instead of writing to the DB directly, we enqueue the write request.
    """
    print(f"Enqueuing message to store: role={role}, content={content}")
    db_queue.put({"job_type": "store_message", "role": role, "content": content})


# 3) Graceful shutdown when your app stops (optional)
def close_db_thread():
    print("Closing DB thread")
    db_queue.put(None)  # Sentinel to break out of the loop
    db_queue.join()
    db_thread.join()

def summarize_and_store(conversation_text):
    print(f"Summarizing and storing conversation text: {conversation_text}")
    summary = call_ollama_to_summarize(conversation_text)
    print(f"Storing summary: {summary}")
    conn = sqlite3.connect("conversation_memory.db", check_same_thread=False)
    c = conn.cursor()
    c.execute("INSERT INTO conversation_memory (summary) VALUES (?)", (summary,))
    conn.commit()
    conn.close()

def call_ollama_to_summarize(conversation_text: str, 
                             model_name: str = "llama3.2",
                             temperature: float = 0.7, 
                             max_tokens: int = 256) -> str:
    """
    Summarize the given `conversation_text` by calling your Ollama /api/chat endpoint
    in streaming mode. Returns the final summary as a string.
    """
    api_url = "https://top-mule-adequately.ngrok-free.app/api/chat"

    # System / user messages for summarization
    messages = [
        {
            "role": "system",
            "content": "You are a concise assistant. You summarize text accurately in a few lines for long-term memory."
        },
        {
            "role": "user",
            "content": (
                "Please summarize the following conversation in a concise manner:\n\n"
                f"{conversation_text}\n\n"
                "Summary:"
            )
        }
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "format": "",  # or "text" if you like; "" means streaming JSON lines
        "options": {
            "temperature": temperature,
            "max_tokens": max_tokens
            
        },
    }

    # We'll read the response line by line, just like in your main code
    summary = ""            # accumulate text here
    start_time = time.time()
    token_count = 0

    try:
        print(f"Calling Ollama API to summarize text: {conversation_text}")
        response = requests.post(api_url, json=payload, stream=True)
        print("Response fetched")
        #print(response)
        for line in response.iter_lines(decode_unicode=True):
            if line.strip():
                # Each line is a JSON object with "message" or "done"
                parsed_line = json.loads(line)
                if "message" in parsed_line and "content" in parsed_line["message"]:
                    chunk = parsed_line["message"]["content"]
                    summary += chunk
                    token_count += len(chunk.split())
                    print(f"Received chunk: {chunk}")

                # If "done": true, we know the stream ended
                if "done" in parsed_line and parsed_line["done"]:
                    break

        elapsed_time = time.time() - start_time
        print(f"Summary generation complete. Time: {elapsed_time:.2f}s, tokens: {token_count}")
    except requests.exceptions.RequestException as e:
        # If there's an API or network error, you can handle it here
        print(f"Error calling Ollama for summary: {e}")
        return ""

    return summary.strip()
