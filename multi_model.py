# Import necessary libraries
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
from json import load
import json
from typing import List
from langchain.document_loaders import PyPDFLoader
import shutil

# URL for the Ollama Llama model API
llamaURL = "http://localhost:11434"

# List of models to use
models = ['llama3.1']

# Template for generating responses based on context and questions
PROMPT_TEMPLATE = """
Context:
{context}

Question:
{question}

Instructions:

    - First, determine whether the context contains sufficient information to answer the question:
        - If yes, respond directly with the answer using the information from the context.
        - If no, explicitly state: "The context does not provide sufficient information to answer this question."
    - In cases where the context is insufficient, follow up immediately with an answer based on general knowledge or reasoning and start it by saying 'this is from external knowledge'.
"""

# Load existing database information from a JSON file
with open('db.json', 'r') as f:
    dataBases = load(f)
print(dataBases)


def add_pdf_to_chromadb(file_path: List[str], dp_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Processes a list of PDF files, splits their content into chunks, and stores them in a ChromaDB database.

    Args:
        file_path (List[str]): List of file paths to the PDFs.
        dp_path (str): Path where the database should be stored.
        chunk_size (int): Maximum size of each text chunk (default is 1000 characters).
        chunk_overlap (int): Overlap between consecutive text chunks (default is 200 characters).
    """
    pdf_paths = file_path  # List of PDF file paths to process
    documents = []  # Container to store documents

    # Load content from each PDF file
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

    # Split loaded text into chunks
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        is_separator_regex=False,
    )
    context = text_spliter.split_documents(documents=documents)

    # Process for each model in the `models` list
    for model in models:
        embedding_model = OllamaEmbeddings(model=model, base_url=llamaURL)

        # Adjust database path to include model name
        splited_dp_path = dp_path.split('/')
        dp_path = '/'.join(splited_dp_path[:-1]) + f'/{model}/' + splited_dp_path[-1]

        # Skip if database already exists
        if os.path.exists(dp_path):
            continue

        # Create and persist database
        db = Chroma(
            persist_directory=dp_path, embedding_function=embedding_model
        )
        db.add_documents(context)
        db.persist()

        # Update database tracking
        if model not in dataBases:
            dataBases[model] = {}
        dataBases[model][splited_dp_path[-1]] = dp_path

    # Save updated database information to a JSON file
    with open('db.json', 'w') as f:
        json.dump(dataBases, f)
    print("Added PDF content to database")


def add_text_to_chromadb(text: str, dp_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Splits plain text into chunks and stores them in a ChromaDB database.

    Args:
        text (str): The input text to be added.
        dp_path (str): Path where the database should be stored.
        chunk_size (int): Maximum size of each text chunk (default is 1000 characters).
        chunk_overlap (int): Overlap between consecutive text chunks (default is 200 characters).
    """
    # Split the input text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)

    # Process for each model in the `models` list
    for model in models:
        embedding_model = OllamaEmbeddings(model=model, base_url=llamaURL)

        # Adjust database path to include model name
        splited_dp_path = dp_path.split('/')
        dp_path = '/'.join(splited_dp_path[:-1]) + f'/{model}/' + splited_dp_path[-1]

        # Skip if database already exists
        if os.path.exists(dp_path):
            continue

        # Create and persist database
        db = Chroma(
            persist_directory=dp_path, embedding_function=embedding_model
        )
        db.add_texts(texts=chunks)
        db.persist()

        # Update database tracking
        if model not in dataBases:
            dataBases[model] = {}
        dataBases[model][splited_dp_path[-1]] = dp_path

    # Save updated database information to a JSON file
    with open('db.json', 'w') as f:
        json.dump(dataBases, f)
    print("Added text to database")


def get_answer(model_name: str, db_path: str, query: str):
    """
    Queries the database using a specified model and retrieves an answer.

    Args:
        model_name (str): Name of the model to use.
        db_path (str): Path to the database to query.
        query (str): The user's query.

    Returns:
        str: The answer generated by the model.
    """
    print('Model:', model_name)
    print('Database path:', db_path)
    print('Query:', query)

    # Initialize embedding model
    embedding_model = OllamaEmbeddings(model=model_name, base_url=llamaURL)

    # Load the database
    db = Chroma(
        persist_directory=db_path, embedding_function=embedding_model
    )

    # Perform a similarity search in the database
    results = db.similarity_search_with_score(query, k=10)
    print('Search results:', results)

    # Create a context from retrieved results
    context = "\n\n----\n\n".join([doc.page_content for doc, _score in results])

    # Generate a prompt using the context and query
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=query)

    # Use the model to generate an answer
    llm = Ollama(model=model_name, base_url=llamaURL, temperature=0.5)
    answer = llm.invoke(prompt)
    return answer


# Streamlit app for the multi-model RAG system
st.title('Multi-Model RAG System')

# Option to add a file to the database
add_file = st.checkbox('Add a file to the database')

if add_file:
    # Choose file type
    file_type = st.selectbox('Select the file type', ['text', 'pdf'])

    if file_type == 'text':
        db_name = st.text_input('Enter the database name')
        text = st.text_area('Enter the text')
        if st.button('Add to database'):
            if text == '' or db_name == '':
                st.error('Please enter the text and the database name')
            else:
                with st.spinner('Adding to database'):
                    add_text_to_chromadb(text, f'rag_env/db/{db_name}')
                st.success('Added to database')

    elif file_type == 'pdf':
        pdf = st.file_uploader('Upload a PDF file')
        if pdf is not None:
            pdf_name = pdf.name
        if st.button('Add to database'):
            if pdf is None:
                st.error('Please upload a PDF file')
            elif pdf_name in dataBases:
                st.error('This PDF is already in the database')
            else:
                with open(f'rag_env/docs/{pdf_name}', 'wb') as f:
                    f.write(pdf.read())
                with st.spinner('Adding to database'):
                    add_pdf_to_chromadb([f'rag_env/docs/{pdf_name}'], f'rag_env/db/{pdf_name}')
                st.success('Added to database')

# Select the model
model_name = st.selectbox('Select the model', models)

if model_name in dataBases:
    db_name = st.selectbox('Select the database', list(dataBases[model_name].keys()))
    if db_name in dataBases[model_name]:
        dp_path = dataBases[model_name][db_name]
    else:
        st.error('Selected database does not exist for the chosen model')
        st.stop()
else:
    st.error('The selected model does not have any processed files. Please add a file to the database.')
    st.stop()

# Input query and get an answer
query = st.text_area('Enter the query')
if st.button('Get the answer'):
    if query == '':
        st.error('Please enter the query')
        st.stop()
    with st.spinner("Searching for the answer..."):
        answer = get_answer(model_name, dp_path, query)
    st.write("Answer: ")
    st.write(answer)
