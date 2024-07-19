# Deep Crawl Assistant

Deep Crawl Assistant is an interactive web application built using Streamlit that allows users to scrape content from multiple URLs, create a knowledge base using Recursive Augmented Generation (RAG), and query the knowledge base using OpenAI's language models. 

## Features

- **URL Scraping:** Scrape content from provided URLs using a parallel scraping mechanism.
- **Knowledge Base Creation:** Generate a knowledge base from scraped content using FAISS for vector storage and retrieval.
- **Interactive Querying:** Query the knowledge base through an interactive chat interface powered by OpenAI's language models.
- **Configuration Flexibility:** Configure various settings such as chunk size, overlap, and model parameters through a settings file and a sidebar interface.

## Installation

### Prerequisites

- Python 3.7 or higher
- Streamlit
- OpenAI API Key

### Clone the Repository

```sh
git clone https://github.com/yourusername/deep-crawl-assistant.git
cd deep-crawl-assistant
```

### Install Dependencies

```sh
pip install -r requirements.txt
```

### Set Up Environment Variables

Create a `.env` file in the root directory and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key
```

## Usage

Run the application using Streamlit:

```sh
streamlit run main.py
```

### Configuration

You can configure the application through the `settings.json` file:

- `model`: The OpenAI model to use (e.g., "gpt-3.5-turbo").
- `top_k`: The number of top results to return from the similarity search.
- `chunk_size`: The size of text chunks to use for the knowledge base.
- `chunk_overlap`: The overlap between text chunks.
- `min_content_length`: The minimum length of content to consider when scraping.
- `max_depth`: The maximum depth to follow links during scraping.

### Sidebar Options

- **OpenAI API Key:** Input your OpenAI API key if not set in the `.env` file.
- **URLs to Scrape:** Provide a list of URLs to scrape, one per line.
- **Scrape and Add to Knowledge Base:** Button to initiate the scraping and knowledge base creation process.
- **Refresh:** Button to refresh the application.

## How It Works

1. **Setup RAG:**
   - Reads and processes JSON files in the current directory.
   - Splits text into chunks using the specified chunk size and overlap.
   - Creates or loads FAISS vector databases for each file.
   
2. **Query RAG:**
   - Merges all individual vector databases into a single database.
   - Performs a similarity search based on the user's query.
   - Returns the most relevant content limited to 4000 characters.

3. **Interactive Chat:**
   - Displays a chat interface where users can interact with the system.
   - Retrieves context from the knowledge base for each user query.
   - Uses OpenAI's language models to generate responses based on the provided context and user query.

## Files

- `main.py`: The main application script.
- `settings.json`: Configuration file for model and other parameters.
- `system_prompt.txt`: The system prompt used by OpenAI's language model.
- `.env`: File to store environment variables, including the OpenAI API key.

## Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- Thanks to the Streamlit community for providing a great framework for building web apps.
- Thanks to OpenAI for their powerful language models.
- Thanks to the developers of FAISS for their efficient similarity search library.

---

Happy Crawling! 🤖📚
