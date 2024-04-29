import os
import pymongo
from dotenv import load_dotenv
from openai import OpenAI
from trulens_eval import Feedback, Select, TruCustomApp, Tru
from trulens_eval.tru_custom_app import instrument
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI as TruLensOpenAI
import numpy as np

tru = Tru()

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

# RAG execution class
class RAG:
    @instrument
    def retrieve(self, query: str) -> list:
        """
        Retrieve relevant text from vector store.
        """
        # Prepare the query text
        query_vector = get_embedding(query)

        # Define the vector search query
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
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
        return list(collection.aggregate(pipeline))

    @instrument
    def generate_completion(self, query: str, context_str: list) -> str:
        """
        Generate answer from context.
        """
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=
        [
            {"role": "user",
            "content": 
            f"We have provided context information below. \n"
            f"---------------------\n"
            f"{context_str}"
            f"\n---------------------\n"
            f"Given this information, please answer the question: {query}"
            }
        ]
        ).choices[0].message.content
        return completion

    @instrument
    def query(self, query: str) -> str:
        context_str = self.retrieve(query)
        completion = self.generate_completion(query, context_str)
        return completion

# Set up TruLens helpers
rag = RAG()
provider = TruLensOpenAI()
grounded = Groundedness(groundedness_provider=provider)

# Define feedback functions
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name = "Groundedness")
    .on(Select.RecordCalls.retrieve.rets.collect())
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name = "Answer Relevance")
    .on(Select.RecordCalls.retrieve.args.query)
    .on_output()
)

f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name = "Context Relevance")
    .on(Select.RecordCalls.retrieve.args.query)
    .on(Select.RecordCalls.retrieve.rets.collect())
    .aggregate(np.mean)
)

# Create Tru app
tru_rag = TruCustomApp(rag,
    app_id = 'RAG v2',
    feedbacks = [f_groundedness, f_answer_relevance, f_context_relevance])

# Tells Tru app to record the execution of this query
with tru_rag as recording:
    rag.query("Recommend a romantic movie suitable for the christmas season and justify your selection")

# Sets up dashboard
tru.get_leaderboard(app_ids=["RAG v1"])
tru.run_dashboard()