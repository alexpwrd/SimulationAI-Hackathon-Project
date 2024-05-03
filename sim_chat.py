import os
import pymongo
import streamlit as st
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
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=key)

# MongoDB setup
mongo_uri = os.getenv("MONGO_URI")
mongo_client = pymongo.MongoClient(mongo_uri)
db = mongo_client["simulation"]
collection = db["synthdata"]

if collection.count_documents({}) == 0:
    raise ValueError("No documents found in MongoDB collection. Please check data population.")

vector_store = MongoDBAtlasVectorSearch(mongo_client, db_name="simulation", collection_name="synthdata", index_name="vector_index", embedding_key="embedding")
index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine(verbose=True)

# TruLens setup
provider = TruLensOpenAI()
context = App.select_context(query_engine)

# Define feedback functions
grounded = Groundedness(groundedness_provider=TruLensOpenAI())
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

# Initialize TruLlama recorder
tru_query_engine_recorder = TruLlama(query_engine, app_id='Simulation_AI', feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance])

st.title('Simulation AI')
query_text = st.text_input("Enter your query (or type 'exit' to quit):")
if st.button('Submit Query'):
    if query_text.lower() == 'exit':
        st.stop()
    with tru_query_engine_recorder as recording:
        response = query_engine.query(query_text)
    response_str = str(response)
    st.write("Response:", response_str)

    result_ids = []
    if response_str.startswith('[') and response_str.endswith(']'):
        try:
            result_ids = eval(response_str)
        except:
            st.error("Error parsing response as list of IDs.")
    elif hasattr(response, 'result_ids'):
        result_ids = [str(id) for id in response.result_ids]

    if result_ids:
        full_documents = collection.find({'_id': {'$in': result_ids}})
        for doc in full_documents:
            st.write("Question:", doc['metadata']['question_text'])
            st.write("Answer:", doc['metadata']['answer'])
            st.write("Other Metadata:", {k: v for k, v in doc['metadata'].items() if k not in ['question_text', 'answer']})

