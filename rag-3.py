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
llamaURL = "http://144.24.204.41:5000"
llm = Ollama(model=model, base_url=llamaURL)
embedding_model = OllamaEmbeddings(model=model, base_url=llamaURL)
db_path='rag_env/db'
db = Chroma(
    persist_directory=db_path,embedding_function=embedding_model
)


st.title("Ask me about hyperParameters")
st.write("please make sure to change the llamaURL to your own URL: http://<your_ip>:<port>")
query = st.text_input("Enter your question here")
if st.button("Get Answer"):
    results = db.similarity_search_with_score(query,k=10)
    context = "\n\n----\n\n".join([doc.page_content for doc,_score in results])

    PROMPT_TEMPLATE ="""
    Answer the question based strictly on the provided context:  
        {context}

    ---

    Respond to the following question based solely on the above context: {question}.  
    - If the answer cannot be determined from the context, explicitly state, "The context does not provide this information."  
    - Afterward, you may provide an answer based on your own knowledge, clearly indicating that it is not derived from the context.

    """

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    prompt = prompt_template.format(context=context,question=query)
    # print('testt',retriever)
    with st.spinner("Searching for the answer..."):
        answer =  llm.invoke(prompt)
    st.write("Answer: ")
    print(answer)
    st.write(answer)
        
