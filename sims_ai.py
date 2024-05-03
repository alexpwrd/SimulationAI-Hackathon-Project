import os
import json
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

def clear_collection(collection):
    collection.delete_many({})
    print("Collection cleared successfully.")

def store_questions(collection, questions):
    if questions:
        try:
            collection.insert_many(questions)
            print("Questions stored successfully.")
        except Exception as e:
            print(f"Error storing questions: {e}")
    else:
        print("No questions to store.")

def fetch_question_by_id(collection, question_id):
    return collection.find_one({"id": question_id})

def store_answer(collection, question_id, answer):
    collection.update_one({"id": question_id}, {"$set": {"answer": answer}})

def generate_questions(scenario, previous_questions):
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
        generated_questions = json.loads(response.choices[0].message.content).get('questions', [])
        print("Generated questions JSON:", generated_questions)
        return generated_questions
    except Exception as e:
        print(f"Failed to generate questions: {e}")
        return []

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
        print(f"Generated Response: {detailed_response}")
        return detailed_response
    except Exception as e:
        print(f"Failed to generate response: {e}")
        return ""

def main():
    print("Welcome to the Simulation Question and Response Generator!")
    scenario = input("Enter the scenario you want to explore: ")
    num_iterations = int(input("How many sets of 10 questions do you want to generate? "))

    collection = connect_to_mongodb()
    clear_collection(collection)

    all_questions = []
    id_counter = 0

    for _ in range(num_iterations):
        questions_json = generate_questions(scenario, all_questions)
        questions_for_db = [{'text': q.get('question', q.get('text', 'No question text provided')), 'id': id_counter + i + 1} for i, q in enumerate(questions_json)]
        
        store_questions(collection, questions_for_db)
        all_questions.extend(questions_for_db)
        id_counter += len(questions_json)

        for question in questions_for_db:
            question_text = question['text']
            detailed_response = generate_detailed_response(question_text)
            store_answer(collection, question['id'], detailed_response)
            print(f"Stored detailed response for question ID {question['id']}")

    print(f"Total questions processed: {len(all_questions)}")
    print("Simulation complete.")

if __name__ == "__main__":
    main()