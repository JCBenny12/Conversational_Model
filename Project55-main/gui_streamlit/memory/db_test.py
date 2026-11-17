from db_write import store_message_in_db_async, db_queue, close_db_thread
import time 

if __name__ == "__main__":
    try:
        # Simulate storing a message
        store_message_in_db_async("user", "Hello, how are you?")
        store_message_in_db_async("assistant", "I'm good, thanks for asking!")
        
        # Simulate summarizing a conversation
        conversation_text = "User: Hello, how are you?\nAssistant: I'm good, thanks for asking!"
        #print("queing summarize job")
        db_queue.put({"job_type": "summarize", "conversation_text": conversation_text})

        # Simulate running for a bit
        time.sleep(3)

    finally:
        # Ensure clean shutdown when script exits
        close_db_thread()