import os
import json
import pymongo
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response.notebook_utils import display_response
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file located in the same directory as the script
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
logging.info("Environment variables loaded.")

# Ensure the OPENAI_API_KEY and MONGO_URI environment variables are set
if "OPENAI_API_KEY" not in os.environ or "MONGO_URI" not in os.environ:
    logging.critical("OPENAI_API_KEY or MONGO_URI not set in environment variables")
    raise EnvironmentError("OPENAI_API_KEY or MONGO_URI not set in environment variables")

# MongoDB setup
mongo_uri = os.getenv("MONGO_URI")
mongo_client = pymongo.MongoClient(mongo_uri)
db = mongo_client["simulation"]
collection = db["synthdata"]

# Fetch data from MongoDB
documents = list(collection.find({}))

# Setup embedding and LLM models
embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=1536)
Settings.embed_model = embed_model

# Ensure that the embedding model is correctly set in the settings
if Settings.embed_model.dimensions != 1536:
    logging.critical("Embedding model dimensions are not set to 1536. Please check the model configuration.")
    raise ValueError("Embedding model dimensions are not set to 1536. Please check the model configuration.")
llm = OpenAI()
Settings.llm = llm
Settings.embed_model = embed_model
logging.info("Embedding and LLM models set up.")

# Convert MongoDB documents to list of Document objects
llama_documents = []
for doc in documents:
    metadata = doc['metadata']
    llama_document = Document(
        text=metadata['question_text'],
        metadata=metadata,
        excluded_llm_metadata_keys=["answer"],
        excluded_embed_metadata_keys=["answer"],
        metadata_template="{key}=>{value}",
        text_template="Question: {content}\nMetadata: {metadata_str}"
    )
    llama_documents.append(llama_document)
logging.info("Documents converted to Llama Document format.")

# Parse documents into nodes and embed
parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(llama_documents)
for node in nodes:
    node_embedding = embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
    node.embedding = node_embedding
logging.info("Documents parsed into nodes and embedded.")

# Clear existing data in MongoDB collection
collection.delete_many({})

# Create and populate vector store
vector_store = MongoDBAtlasVectorSearch(mongo_client, db_name="simulation", collection_name="synthdata", index_name="vector_index")
vector_store.add(nodes)
logging.info("Vector store created and populated.")