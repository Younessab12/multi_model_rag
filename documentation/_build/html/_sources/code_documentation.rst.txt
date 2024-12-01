Code Documentation
==================

File Overview
-------------

**multi_model.py**
This is the main entry point of the application. It integrates multiple LLMs, a Chroma vector database, and a Streamlit interface to provide a Retrieval-Augmented Generation (RAG) system. It allows users to query processed documents or add new files to the database for processing.

Modules and Functions
---------------------

1. **add_pdf_to_chromadb**:
   
   - Processes PDF documents by splitting them into chunks and storing them as embeddings in the Chroma vector database.
   - Supports multi-model embedding by storing embeddings specific to each model in its respective directory.

2. **add_text_to_chromadb**:
   
   - Processes raw text by splitting it into chunks and storing embeddings in the Chroma vector database.
   - Similar to PDFs, supports multi-model embeddings.

3. **get_answer**:
   
   - Fetches relevant context from the Chroma database using similarity search and combines it with the user query to generate a response.
   - Uses a predefined prompt template for consistent formatting of the answer.

4. **Streamlit Interface**:
   
   - **File Upload**:
  
     - Users can add text or PDF files to the database.
     - Provides options for specifying the file type and database name.
   - **Model Selection**:
  
     - Allows users to choose the model for embedding and querying.
   - **Database Selection**:
  
     - Users can select the database corresponding to the chosen model.
   - **Query Input**:
  
     - Users can input their questions and retrieve answers based on the selected database and model.

Flow Overview
-------------

1. **Adding Files**:
   
   - Users can add text or PDF files to the database.
   - The application processes the file, splits it into chunks, creates embeddings, and stores them in the Chroma vector database.

2. **Selecting Model and Database**:
   
   - Users select the model and the corresponding database to use for retrieval.

3. **Querying**:

   - Users enter a question.
   - The app retrieves contextually relevant chunks from the selected database.
   - If the context is sufficient, the app generates an answer based solely on the context. Otherwise, it generates an answer based on the model’s external knowledge.

4. **Answer Display**:
   
   - The app displays the answer along with its origin (context-based or external).

Dependencies
------------

- **langchain**:
  
  - Used for creating embeddings and managing retrieval chains.
- **streamlit**:
  
  - Provides an interactive web-based user interface.

- **Ollama**:
  
  - Integrates LLMs for embedding and answer generation.
- **Chroma**:
  
  - Serves as the vector database for storing and retrieving document embeddings.
- **PyPDFLoader**:
  
  - Loads and processes PDF documents.
- **RecursiveCharacterTextSplitter**:
  
  - Splits text or document content into manageable chunks for embedding.
