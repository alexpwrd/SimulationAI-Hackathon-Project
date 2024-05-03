import os
from dotenv import load_dotenv
from pymongo import MongoClient
from openai import OpenAI

# Load environment variables
load_dotenv()

# Setup OpenAI client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# MongoDB setup
def connect_to_mongodb():
    uri = os.getenv("MONGO_URI")
    mongo_client = MongoClient(uri)
    db = mongo_client["simulation"]
    return db["synthdata"]

def fetch_question_by_id(collection, question_id):
    return collection.find_one({"id": question_id})

def generate_detailed_response(question):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a knowledgeable AI tasked with simulating the most likely outcomes for the following scenario. Describe a cascade of outcomes given the question. Provide a highly detailed answer, utilizing your maximum context window of 4096 tokens to ensure a comprehensive exploration of the topic, that's around 200 sentences or 40 paragraphs."},
                {"role": "user", "content": question}
            ],
            max_tokens=4096
        )
        detailed_response = response.choices[0].message.content.strip()
        print(f"Generated Response: {detailed_response}")  # Log the response to the console
        return detailed_response
    except Exception as e:
        print(f"Failed to generate response: {e}")
        return ""

def store_answer(collection, question_id, answer):
    collection.update_one({"id": question_id}, {"$set": {"answer": answer}})

def main():
    collection = connect_to_mongodb()
    question_id = 1  # Start with the first question
    while True:
        question = fetch_question_by_id(collection, question_id)
        if not question:
            break  # Break the loop if there are no more questions
        question_text = question['text']
        detailed_response = generate_detailed_response(question_text)
        store_answer(collection, question_id, detailed_response)
        print(f"Stored detailed response for question ID {question_id}")
        question_id += 1  # Increment to fetch the next question

if __name__ == "__main__":
    main()