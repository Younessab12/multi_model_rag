Installation
============

Prerequisites
-------------

 - Python 3.8 or higher.
 - **Ollama**: The LLM server for running models locally. Download and install Ollama from [https://ollama.com/download].
 - Dependencies listed in `requirements.txt`.

Setup Instructions
------------------

1. Clone the repository:
    .. code-block:: bash

      git clone https://github.com/Younessab12/multi_model_rag
      cd multi_model_rag

2. Create a virtual environment named `rag_env`:
    .. code-block:: bash

      python -m venv rag_env

3. Activate the virtual environment:
   - On Windows:
  
      .. code-block:: bash

         rag_env\Scripts\activate

   - On macOS/Linux:
  
      .. code-block:: bash

         source rag_env/bin/activate

4. Install dependencies:
    .. code-block:: bash

      pip install -r requirements.txt

5. Download and install **Ollama**:
   
   Follow the installation guide at [https://ollama.com/download].

6. Pull available models from Ollama:
   
   After installing Ollama, you can pull models such as `llama3.1` and `mistral` with the following commands:
    .. code-block:: bash

      ollama pull llama3.1
      ollama pull mistral
      ollama pull 0ssamaak0/silma-v1
      ollama pull llama3.2
      ollama pull aya-expanse

   This ensures the models are available locally for the application.

7. Update `multi_model.py`:
   
   Open the `multi_model.py` file and configure the following variables directly:

   - `llamaURL`: Set this to your Ollama server URL (e.g., `http://localhost:11434`).

8. Start the application:
    .. code-block:: bash

      streamlit run multi_model.py
