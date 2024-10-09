# Open Notebook

Open Notebook is an AI-powered knowledge management and question-answering system built with Streamlit. It allows users to create a personalized knowledge base from various sources and interact with it using natural language queries.

## Features

- **AI-Powered Conversations**: Utilizes OpenAI's GPT models for intelligent responses.
- **Custom Knowledge Base**: Add content from websites, PDFs, and custom text inputs.
- **RAG (Retrieval-Augmented Generation)**: Enhances AI responses with relevant information from your knowledge base.
- **User-Friendly Interface**: Clean, dark-themed UI with expandable sections for easy navigation.
- **Flexible Configuration**: Customize AI model, retrieval parameters, and more.

## Installation

1. Clone the repository:
   ```
   https://github.com/mshojaei77/open-notebook.git
   ```
   ```
   cd open-notebook
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Create a `.env` file in the project root.
   - Add your API key: `OPENAI_API_KEY=your_api_key_here`

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the provided local URL (usually `http://localhost:8501`).

3. Configure the app:
   - Enter your OpenAI API key if not set in the `.env` file.
   - Adjust advanced settings like AI model and chunk size if needed.

4. Build your knowledge base:
   - Add websites by entering URLs.
   - Upload PDF documents.
   - Input custom text directly.

5. Start asking questions! The AI will respond based on your knowledge base.

## Configuration

- **AI Model**: Choose between different GPT models.
- **Top K**: Number of relevant documents to retrieve for each query.
- **Chunk Size**: Size of text chunks for processing.
- **Chunk Overlap**: Overlap between text chunks.

## Managing Your Knowledge Base

- View all items in your knowledge base.
- Remove individual items or clear the entire knowledge base.
- Refresh the app to see updates.

## File Structure

- `app.py`: Main application file.
- `crawler.py`: Web scraping functionality.
- `system_prompt.txt`: System prompt for the AI.
- `knowledge_base/`: Directory for storing knowledge base files.
  - `json/`: JSON files of processed content.
  - `faiss/`: FAISS vector stores for efficient retrieval.

## Dependencies

- Streamlit
- LangChain
- OpenAI
- FAISS
- PyPDF2
- python-dotenv

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the GPT models.
- Streamlit for the web app framework.
- LangChain for RAG implementation.
