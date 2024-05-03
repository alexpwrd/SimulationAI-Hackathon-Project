import os
from dotenv import load_dotenv
from pymongo import MongoClient
from openai import OpenAI
import json

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

def clear_collection(collection):
    collection.delete_many({})
    print("Collection cleared successfully.")

def store_questions(collection, questions):
    if questions:  # Check if the questions list is not empty
        try:
            collection.insert_many(questions)
            print("Questions stored successfully.")
        except Exception as e:
            print(f"Error storing questions: {e}")
    else:
        print("No questions to store.")

def generate_questions(scenario, previous_questions):
    # Compile context from previous questions
    prior_context = " ".join([q['text'] for q in previous_questions]) if previous_questions else ""
    user_prompt = f"Considering these previous questions: {prior_context} Now, generate a list of 10 new questions about the consequences of {scenario}, focusing on both immediate and long-term impacts in JSON format." if prior_context else f"Generate a list of 10 new questions about the consequences of {scenario} in JSON format, covering both immediate and long-term impacts."

    messages = [
        {"role": "system", "content": "You are a helpful AI tasked with generating insightful questions in JSON format."},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            response_format={"type": "json_object"}
        )
        # Correctly accessing the response content
        generated_questions = json.loads(response.choices[0].message.content).get('questions', [])
        print("Generated questions JSON:", generated_questions)
        return generated_questions
    except Exception as e:
        print(f"Failed to generate questions: {e}")
        return []

def main():
    print("Welcome to the Simulation Question Generator!")
    scenario = input("Enter the scenario you want to explore: ")
    num_iterations = int(input("How many sets of 10 questions do you want to generate? "))
    
    collection = connect_to_mongodb()
    clear_collection(collection)
    
    all_questions = []
    id_counter = 0  # Initialize a counter for question IDs

    for _ in range(num_iterations):
        questions_json = generate_questions(scenario, all_questions)
        # Update IDs to be cumulative and handle key inconsistencies
        questions_for_db = [{
            'text': q.get('question', q.get('text', 'No question text provided')),  # Check for 'question' or 'text' key
            'id': id_counter + i + 1
        } for i, q in enumerate(questions_json)]
        
        store_questions(collection, questions_for_db)
        all_questions.extend(questions_for_db)  # Append directly the questions in the same format for context preparation
        id_counter += len(questions_json)  # Update the counter based on the number of questions generated
        
        print("\nGenerated Questions:")
        for question in questions_for_db:
            print(f"- {question['text']}")

    print(f"Total questions generated: {len(all_questions)}")

if __name__ == "__main__":
    main()


