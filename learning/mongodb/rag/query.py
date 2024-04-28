import os
import pymongo
from dotenv import load_dotenv
import pprint
from openai import OpenAI

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# MongoDB setup
mongo_uri = os.getenv("MONGO_URI")
mongo_client = pymongo.MongoClient(mongo_uri)
db = mongo_client["movies"]
collection = db["movies_records"]

# Ensure the collection is not empty
if collection.count_documents({}) == 0:
    raise ValueError("No documents found in MongoDB collection. Please check data population.")

# OpenAI setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Prepare the query text
query_text = "Recommend a romantic movie suitable for the christmas season and justify your selection"
query_vector = get_embedding(query_text)

# Define the vector search query
pipeline = [
    {
        "$vectorSearch": {
            "index": "old_vectorindex",
            "path": "embedding",
            "queryVector": query_vector,
            "numCandidates": 100,
            "limit": 3
        }
    },
    {
        "$project": {
            "_id": 0,
            "id": 1,
            "title": "$metadata.title",
            "description": "$metadata.fullplot",
            "score": {"$meta": "vectorSearchScore"},
            "genres": "$metadata.genres",
            "runtime": "$metadata.runtime",
            "imdb": "$metadata.imdb",
            "directors": "$metadata.directors",
            "countries": "$metadata.countries",
            "awards": "$metadata.awards",
            "languages": "$metadata.languages",
            "cast": "$metadata.cast",
            "poster": "$metadata.poster"
        }
    }
]

# Execute the aggregation pipeline
try:
    results = list(collection.aggregate(pipeline))
    pprint.pprint(results)
    # Format results for sending to ChatGPT
    formatted_results = "\n".join([f"Title: {movie['title']}, Description: {movie['description']}" for movie in results])

    # Prepare messages for the chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query_text},
        {"role": "assistant", "content": f"The top 3 movie recommendations are:\n{formatted_results}"},
        {"role": "user", "content": "Can you summarize the results of this query?"}
    ]

    # Call the Chat Completions API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    # Extract the response content
    if response.choices:
        assistant_message = response.choices[0].message.content
        print("OpenAI Assistant says:", assistant_message)
    else:
        print("No response from OpenAI Assistant.")

except Exception as e:
    print(f"An error occurred: {e}")