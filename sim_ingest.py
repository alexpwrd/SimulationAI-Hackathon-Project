import os
import pymongo
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# MongoDB setup
mongo_uri = os.getenv("MONGO_URI")
mongo_client = pymongo.MongoClient(mongo_uri)
db = mongo_client["simulation"]
synthdata_collection = db["synthdata"]
synthdata_embedding_collection = db["synthdata_embedding"]

# Clear existing data in the embedding collection
synthdata_embedding_collection.delete_many({})

# Setup embedding model
embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=os.getenv('OPENAI_API_KEY'))

# Fetch data from synthdata collection
documents = synthdata_collection.find({})

# Process each document
for doc in documents:
    text = doc["text"] + " " + doc["answer"]  # Combine question and answer for embedding
    embedding = embed_model.get_text_embedding(text)

    # Create a new document with the original data and the embedding
    new_doc = {
        "text": doc["text"],
        "answer": doc["answer"],
        "embedding": embedding,  # No need to convert to list, as it's already a list
        "metadata": {}  # Adding a dummy metadata field
    }
    synthdata_embedding_collection.insert_one(new_doc)

print("Data ingested and embeddings stored in 'synthdata_embedding' collection.")