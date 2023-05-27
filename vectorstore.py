# Import os to set API key
import os
from apikey import apikey
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
# Bring in streamlit for UI/app interface
import streamlit as st

# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import DeepLake as the vector store 
from langchain.vectorstores import DeepLake
from langchain.embeddings import OpenAIEmbeddings

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# Set APIkey for OpenAI Service
# Can sub this out for other LLM providers
os.environ['OPENAI_API_KEY'] = apikey

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)

# Create and load PDF Loader
loader = PyPDFLoader('COA-Motorized-owners-manual.pdf')
# Split pages from pdf 
pages = loader.load_and_split()
# Load documents into vector database aka ChromaDB
#store = Chroma.from_documents(pages, collection_name='annualreport')

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'

embedding = OpenAIEmbeddings()
vectordb = DeepLake.from_documents(documents=pages, embedding=embedding,dataset_path="./my_deeplake/",overwrite=True)

vectordb.persist()
vectordb = None
