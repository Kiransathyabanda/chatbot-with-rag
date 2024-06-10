# Chat and Document Analysis App

This is a Streamlit web application that allows users to chat with a chatbot or analyze documents for answers to their questions. The app uses LangChain and Groq APIs for advanced language understanding and text analysis.

## Features

- **Chat Mode**: Chat with a chatbot that uses advanced language models to provide answers and information.
- **Document Analysis Mode**: Upload PDFs and ask questions related to the content of the documents.
- **Conversational Memory**: The chatbot can remember the context of the conversation for better responses.
- **Customizable Models**: Choose from different language models for the chatbot.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/chat-doc-analysis-app.git
    cd chat-doc-analysis-app
    ```

2. **Create and activate a virtual environment** (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    - Create a `.env` file in the root directory.
    - Add your Groq API key to the `.env` file:
      ```
      GROQ_API_KEY=your_groq_api_key
      ```

## Usage

1. **Run the Streamlit app**:
    ```sh
    streamlit run app.py
    ```

2. **Navigate to the app in your browser**:
    - The app will typically be running at `http://localhost:8501`.

3. **Interact with the app**:
    - Use the sidebar to switch between Chat Mode and Document Analysis Mode.
    - Upload PDF documents and ask questions about their content.
    - Chat with the chatbot for various queries.

## Files

- `app.py`: Main Streamlit application file.
- `requirements.txt`: List of Python dependencies.
- `.env`: File for environment variables (not included in the repository, needs to be created by the user).

## Dependencies

- Streamlit
- LangChain
- PyPDF2
- FAISS
- Groq API
- dotenv

## Credits

- Developed by [kiransathyabanda]

## License

This project is licensed under the [MIT License](LICENSE).
