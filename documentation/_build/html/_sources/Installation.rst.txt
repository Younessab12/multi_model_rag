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

      git clone <repository_url>
      cd <repository_directory>

2. Install dependencies:
    .. code-block:: bash

      pip install -r requirements.txt

3. Download and install **Ollama**:
   
   Follow the installation guide at [https://ollama.com/download].

4. Pull available models from Ollama:
   
   After installing Ollama, you can pull models such as `llama3.1` and `mistral` with the following commands:
    .. code-block:: bash

      ollama pull llama3.1
      ollama pull mistral

   This ensures the models are available locally for the application.

5. Update `multi_model.py`:
   
   Open the `multi_model.py` file and configure the following variables directly:

- `llamaURL`: Set this to your Ollama server URL (e.g., `http://localhost:11434`).

6. Start the application:
    .. code-block:: bash

      streamlit run multi_model.py
