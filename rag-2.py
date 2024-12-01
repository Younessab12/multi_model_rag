import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st



model = 'llama3.1'
# llamaURL = "http://172.20.221.171:11434"
llamaURL = "http://localhost:11434"
llm = Ollama(model=model, base_url=llamaURL)
embedding_model = OllamaEmbeddings(model=model, base_url=llamaURL)
db_path = 'rag_env/db'

vector_store = Chroma(persist_directory=db_path,embedding_function=embedding_model)

retriever = vector_store.as_retriever(search_kwargs={'k': 5})

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain = create_stuff_documents_chain(llm,retrieval_qa_chat_prompt)

retrieval_chain = create_retrieval_chain(retriever,combine_docs_chain)

st.title("Ask me about hyperParameters")
st.write("please make sure to change the llamaURL to your own URL: http://<your_ip>:<port>")
qst = st.text_input("Enter your question here")
if st.button("Get Answer"):
    with st.spinner("Searching for the answer..."):
        answer =  retrieval_chain.invoke({"input":qst})
    st.write("Answer: ")
    st.write(answer["answer"])
        
