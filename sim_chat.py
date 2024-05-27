import os
import pymongo
import streamlit as st
from dotenv import load_dotenv

from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# Set Streamlit page configuration
st.set_page_config(page_title="Simulation AI Chat")
st.title("Simulation AI Chat")

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

# Ensure documents are correctly embedded
vector_store = MongoDBAtlasVectorSearch(mongo_client, db_name="simulation", collection_name="synthdata", index_name="vector_index", embedding_key="embedding")
index = VectorStoreIndex.from_vector_store(vector_store)

query_engine = index.as_query_engine(verbose=True)

# Initialize the chat messages history
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I assist you today?"}
    ]

# Function to add a new message to the chat history
def add_message(sender, message):
    st.session_state.messages.append({"role": sender, "content": message})

# Display the chat messages history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
if prompt := st.chat_input("Enter your query (or type 'exit' to quit):"):
    st.write('prompt:', prompt)
    add_message("user", prompt)

    if prompt.lower() == "exit":
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Send the query to the AI and get the response
            response = query_engine.query(prompt)

            if response:
                try:
                    ai_response = response.response  # Accessing the response text
                    if ai_response == 'Empty Response':
                        st.write("No meaningful response from the AI.")
                        add_message("assistant", "Sorry, I couldn't get a meaningful response for that query.")
                    else:
                        st.write(ai_response)
                        add_message("assistant", ai_response)

                    # Retrieve and display relevant documents from source_nodes
                    if response.source_nodes:
                        for node in response.source_nodes:
                            with st.expander(f"Document ID: {node.id_}"):
                                st.write(f"**Text:** {node.text}")
                                st.write(f"**Score:** {node.metadata.get('score', 'N/A')}")
                                st.write(f"**Metadata:** {node.metadata}")
                    else:
                        st.write("No relevant documents found.")
                except AttributeError as e:
                    st.write("Error retrieving response:", str(e))
                    add_message("assistant", "Sorry, I couldn't process that request.")
            else:
                st.write("Received an empty response from the query engine.")
                add_message("assistant", "Sorry, I couldn't get a response for that query.")
