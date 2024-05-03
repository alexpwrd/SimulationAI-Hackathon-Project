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

# Setup embedding model
embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=os.getenv('OPENAI_API_KEY'))

# Fetch data from synthdata collection
documents = synthdata_collection.find({})

# Process each document
for doc in documents:
    text = doc["text"] + " " + doc["answer"]  # Combine question and answer for embedding
    embedding = embed_model.get_text_embedding(text)

    # Update the existing document with the embedding
    synthdata_collection.update_one(
        {"_id": doc["_id"]},
        {"$set": {"embedding": embedding}}
    )

print("Embeddings added to existing documents in 'synthdata' collection.")