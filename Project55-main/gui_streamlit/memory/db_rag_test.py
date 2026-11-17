from db_rag import fetch_data_from_db, store_embeddings, retrieve_context

if __name__ == "__main__":
    print("Fetching data from database...")
    data = fetch_data_from_db()

    if not data:
        print("No data found in the database. Exiting.")
    else:
        print(f"Fetched {len(data)} records. Storing embeddings...")

        store_embeddings(data)
        print("Embeddings stored successfully!")

        # Test retrieval
        test_query = "Tell me about a recent conversation."
        print(f"Testing retrieval for query: {test_query}")

        retrieved_context = retrieve_context(test_query)

        print("\nRetrieved Context:")
        for i, doc in enumerate(retrieved_context):
            print(f"{i+1}. {doc}")
