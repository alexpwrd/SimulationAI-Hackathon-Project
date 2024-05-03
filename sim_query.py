#sim_query.py

import os
import pymongoe
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

# Set the global default embedding model
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=key)

# MongoDB setup
mongo_uri = os.getenv("MONGO_URI")
mongo_client = pymongo.MongoClient(mongo_uri)
db = mongo_client["simulation"]
collection = db["synthdata"]

# Ensure the collection is not empty
if collection.count_documents({}) == 0:
    raise ValueError("No documents found in MongoDB collection. Please check data population.")

# Initialize vector store for the simulation data
vector_store = MongoDBAtlasVectorSearch(mongo_client, db_name="simulation", collection_name="synthdata", index_name="vector_index", embedding_key="embedding")
index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine(verbose=True)

# TruLens setup
provider = TruLensOpenAI()
context = App.select_context(query_engine)

# Feedback functions
grounded = Groundedness(groundedness_provider=provider)
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons)
    .on(context.collect())
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

f_answer_relevance = (
    Feedback(provider.relevance)
    .on_input_output()
)
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons)
    .on_input()
    .on(context)
    .aggregate(np.mean)
)

# Using TruLlama to record feedback during queries
with TruLlama(query_engine, app_id='AI_Simulator', feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance]) as recording:
    # Perform a vector-based query
    result = query_engine.query("What would be the immediate impact of World War 3 on global supply chains?")

# Display results and run feedback analysis if needed
print("Query Results:", result)

from trulens_eval import Tru
tru = Tru()
tru.run_dashboard()