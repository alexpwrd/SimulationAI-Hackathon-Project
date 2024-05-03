import os
import pymongo
from dotenv import load_dotenv
import numpy as np

from trulens_eval import Feedback, TruLlama
from trulens_eval.feedback import Groundedness
from trulens_eval.app import App
from trulens_eval.feedback.provider.openai import OpenAI as TruLensOpenAI

from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

key = os.getenv('OPENAI_API_KEY')

# Global default
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=key)

# MongoDB setup
mongo_uri = os.getenv("MONGO_URI")
mongo_client = pymongo.MongoClient(mongo_uri)
db = mongo_client["simulation"]
collection = db["synthdata"]

# Ensure the collection is not empty
if collection.count_documents({}) == 0:
    raise ValueError("No documents found in MongoDB collection. Please check data population.")

# Initialize vector store
vector_store = MongoDBAtlasVectorSearch(mongo_client, db_name="simulation", collection_name="synthdata", index_name="vector_index", embedding_key="embedding")
index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine(verbose=True)

# TruLens setup
provider = TruLensOpenAI()
context = App.select_context(query_engine)

# Define a groundedness feedback function
grounded = Groundedness(groundedness_provider=TruLensOpenAI())
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons)
    .on(context.collect())  # Collect context chunks into a list
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

# Question/answer relevance between overall question and answer
f_answer_relevance = (
    Feedback(provider.relevance)
    .on_input_output()
)

# Question/statement relevance between question and each context chunk
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons)
    .on_input()
    .on(context)
    .aggregate(np.mean)
)

# Initialize TruLlama recorder
tru_query_engine_recorder = TruLlama(query_engine,
    app_id='Simulation_AI',
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance])

# Using context manager for query execution
with tru_query_engine_recorder as recording:
    while True:
        # User input for query
        query_text = input("Enter your query (or type 'exit' to quit): ")
        if query_text.lower() == 'exit':
            break
        response = query_engine.query(query_text)

        # Convert any response type to string
        response_str = str(response)

        # Now handle the string response
        print("Response:", response_str)
        result_ids = []  # Update or process result_ids if needed based on the response_str

        # Assuming the response might contain IDs or further actionable data
        if response_str.startswith('[') and response_str.endswith(']'):
            # Try to parse as list of IDs if response looks like a list
            try:
                result_ids = eval(response_str)
            except:
                print("Error parsing response as list of IDs.")
        elif hasattr(response, 'result_ids'):
            # Handling response objects with a 'result_ids' attribute
            result_ids = [str(id) for id in response.result_ids]
        else:
            # Handle as plain text or log if needed
            print("Handled as plain text response or log accordingly.")

        if result_ids:
            # Fetch full documents based on result IDs
            full_documents = collection.find({'_id': {'$in': result_ids}})

            # Process and display results
            for doc in full_documents:
                print("Question:", doc['metadata']['question_text'])
                print("Answer:", doc['metadata']['answer'])
                print("Other Metadata:", {k: v for k, v in doc['metadata'].items() if k not in ['question_text', 'answer']})
        else:
            print("")

from trulens_eval import Tru
tru = Tru()
tru.run_dashboard()