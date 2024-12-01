Usage
=====

Running the Application
------------------------

Start the application:
   .. code-block:: bash

      streamlit run multi_model.py

Interactive Features
--------------------

1. **Process a New File**:
   
   - Check the checkbox labeled **"Add file to database"** if you want to process and add a new file to the database.
   - If the checkbox is selected:
     - You will see a new field where you can select the document type: **PDF** or **Text**.
     - For text files, enter the text directly. For PDFs, upload the file.
     - Click the button labeled **"Add to Database"** to process the file and add it to the Chroma database.

2. **Select the Model**:
   
   - Choose the model you want to use from the list of available models, such as `llama3.1` or `mistral`.

3. **Select the Database**:
   
   - Pick the database you want to use for retrieving context.

4. **Enter Your Query**:
   
   - Type your question in the text input box.
   - Click the **"Get the Answer"** button to retrieve the response.

**Answer Behavior**:

- If the database contains the context for your query, the answer will be entirely based on the context.
- If the context is not available, the app will indicate that the context does not provide the answer and generate a response based on the model's knowledge.

Tips:

- Ensure the vector database is properly initialized.
- Keep your model URL (`llamaURL`) correctly configured in the application code.
