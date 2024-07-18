# Deep Crawl RAG System

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Configuration](#configuration)
7. [How It Works](#how-it-works)
8. [Troubleshooting](#troubleshooting)
9. [Contributing](#contributing)
10. [License](#license)

## Introduction

Deep Crawl RAG System is an advanced web scraping and question-answering tool that leverages the power of Retrieval-Augmented Generation (RAG) to provide intelligent responses based on scraped web content. This system allows users to input multiple URLs, scrape their content in parallel, and then use this information to answer user queries.
![image](https://github.com/user-attachments/assets/5e367344-775b-43ed-9953-d87ecd1aed91)

## Features

- Multi-URL web scraping with customizable depth and content quality filters
- Parallel processing for efficient scraping of multiple websites
- Automatic text cleaning and preprocessing
- Document chunking for optimal information retrieval
- Integration with OpenAI's language models for intelligent answer generation
- User-friendly Streamlit interface for easy interaction
- Customizable configuration options for scraping and RAG setup

## Requirements

- Python 3.7+
- Streamlit
- Scrapy
- BeautifulSoup4
- LangChain
- OpenAI API
- FAISS

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/mshojaei77/Deep-Crawl.git
   cd Deep-Crawl
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run Chat.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Enter your OpenAI API key in the sidebar.

4. Input the URLs you want to scrape (one per line) in the sidebar.

5. Adjust the scraping and RAG settings as needed in `Settings' Page.

6. Click "Scrape and Setup RAG" to start the process.

7. Once the system is ready, enter your question in the main area and click "Get Answer".

## Configuration

The sidebar provides several configuration options:

- **OpenAI API Key**: Your API key for accessing OpenAI's models.
- **URLs to scrape**: Enter one URL per line.
- **Max crawling depth**: Set the maximum depth for web crawling (1-10).
- **Min content length**: Set the minimum length for content to be considered valid (50-500 characters).
- **Chunk size**: Set the size of text chunks for processing (500-2000 characters).
- **Chunk overlap**: Set the overlap between chunks (0-200 characters).
- **Number of similar documents**: Set the number of documents to retrieve for each query (1-10).
- **OpenAI Model**: Choose between "gpt-3.5-turbo" and "gpt-4".

## How It Works

1. **Web Scraping**: The system uses Scrapy to crawl the provided URLs up to the specified depth. It cleans the HTML content and filters out low-quality or irrelevant pages.

2. **Text Processing**: The scraped content is split into chunks using LangChain's RecursiveCharacterTextSplitter.

3. **Embedding**: The text chunks are embedded using OpenAI's embedding model.

4. **Indexing**: The embeddings are indexed using FAISS for efficient similarity search.

5. **Query Processing**: When a user asks a question, the system performs a similarity search to find the most relevant text chunks.

6. **Answer Generation**: The relevant chunks are sent to OpenAI's language model along with the user's question to generate a comprehensive answer.

## Troubleshooting

- If you encounter any errors related to the OpenAI API, make sure your API key is correct and you have sufficient credits.
- If the scraping process is slow, try reducing the max crawling depth or the number of URLs.
- If you're not getting relevant answers, try adjusting the chunk size, chunk overlap, or the number of similar documents.

## Contributing

Contributions to improve Deep Crawl RAG System are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes with clear, descriptive messages.
4. Push your branch and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
