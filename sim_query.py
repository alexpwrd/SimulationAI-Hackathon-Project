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

# global default
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

# Execute query
# result = query_engine.query("What are the potential long-term effects of COVID-19 on global health?")

# TruLens

provider = TruLensOpenAI()

# select context to be used in feedback. the location of context is app specific.
context = App.select_context(query_engine)

# Define a groundedness feedback function
grounded = Groundedness(groundedness_provider=TruLensOpenAI())
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons)
    .on(context.collect()) # collect context chunks into a list
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

# Question/answer relevance between overall question and answer.
f_answer_relevance = (
    Feedback(provider.relevance)
    .on_input_output()
)
# Question/statement relevance between question and each context chunk.
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons)
    .on_input()
    .on(context)
    .aggregate(np.mean)
)

tru_query_engine_recorder = TruLlama(query_engine,
    app_id='Simulation_AI',
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance])

# or as context manager
with tru_query_engine_recorder as recording:
    query_engine.query("What are the immediate measures recommended by health organizations in response to COVID-20?")

from trulens_eval import Tru

tru = Tru()

tru.run_dashboard()