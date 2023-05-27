import os
from apikey import apikey

import streamlit as st 
from langchain.llms import OpenAI,OpenAIChat
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain,RetrievalQA
from langchain.memory import ConversationBufferMemory

# Import DeepLake as the vector store 
from langchain.vectorstores import DeepLake
from langchain.embeddings import OpenAIEmbeddings

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

os.environ['OPENAI_API_KEY'] = apikey

embeddings = OpenAIEmbeddings()
store = DeepLake(dataset_path="./my_deeplake/", embedding_function=embeddings, read_only=True)

# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name="rv manual",
    description="rv manual",
    vectorstore=store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

#query = "how to access engine compartment"
#docs = db.similarity_search(query)
#qa = RetrievalQA.from_chain_type(llm=OpenAIChat(model='gpt-3.5-turbo'), chain_type='stuff', retriever=db.as_retriever())
#print(qa.run(query))

st.title('RV help bot')
# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = agent_executor.run(prompt)
    # ...and write it out to the screen
    st.write(response)

    # With a streamlit expander  
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt) 
        # Write out the first 
        st.write(search[0][0].page_content)
