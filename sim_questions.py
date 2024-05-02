import os
from dotenv import load_dotenv
from pymongo import MongoClient
from openai import OpenAI
import json

# Load environment variables
load_dotenv()

print("URI:", os.getenv("MONGO_URI"))  # Debugging

# Setup OpenAI client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# MongoDB setup
def connect_to_mongodb():
    uri = os.getenv("MONGO_URI")
    mongo_client = MongoClient(uri)
    db = mongo_client["simulation"]
    return db["synthdata"]

def generate_questions(scenario):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": f"Generate a list of 10 questions about the consequences of {scenario}, covering immediate and long-term impacts."}
        ],
        response_format={"type": "json_object"}
    )
    print("Generated JSON:", response.choices[0].message.content)  # Debugging line to check the JSON output
    return response.choices[0].message.content

def store_questions(collection, questions_json):
    questions_data = json.loads(questions_json)
    # Access the list of questions directly
    questions_list = questions_data.get('questions', [])
    if not isinstance(questions_list, list) or not questions_list:
        print("Error: Expected a non-empty list of dictionaries. Received:", questions_list)
        return  # Exit if the data is not as expected
    
    # Convert list of strings to list of dictionaries
    questions_dict_list = [{'question': question} for question in questions_list]

    collection.insert_many(questions_dict_list)
    print("Questions stored successfully in MongoDB.")
    
def main():
    print("Welcome to the Simulation Question Generator!")
    scenario = input("Enter the scenario you want to explore: ")
    collection = connect_to_mongodb()
    questions_json = generate_questions(scenario)
    if questions_json:
        store_questions(collection, questions_json)
        questions = json.loads(questions_json)
        print("\nGenerated Questions:")
        for question in questions:
            print(f"- {question}")
    else:
        print("No valid questions generated.")

if __name__ == "__main__":
    main()