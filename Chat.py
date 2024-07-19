import streamlit as st
import json
import os
from pathlib import Path
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from crawler import scrape_urls_parallel

def setup_rag(chunk_size, chunk_overlap, files):
    if 'api_key' not in st.session_state or not st.session_state.api_key:
        st.error("Please enter your OpenAI API Key in the sidebar.")
        return None

    try:
        embeddings = OpenAIEmbeddings(api_key=st.session_state.api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI Embeddings: {str(e)}")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n'], chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    parent_dir = os.path.abspath(os.path.join(os.getcwd()))
    vector_dbs = {}

    for file in files:
        if file.endswith(".json") and file != "settings.json":
            try:
                file_path = os.path.join(parent_dir, file)
                embedding_file = Path(file_path.replace(".json", ".faiss"))

                if embedding_file.exists():
                    vector_db = FAISS.load_local(
                        str(embedding_file), embeddings, allow_dangerous_deserialization=True)
                else:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    text = ' '.join([item['content'] for item in data])
                    documents = text_splitter.create_documents([text])
                    vector_db = FAISS.from_documents(documents, embeddings)
                    vector_db.save_local(str(embedding_file))

                vector_dbs[file] = vector_db
            except Exception as e:
                st.error(f"Error processing file {file}: {str(e)}")
                continue

    return vector_dbs

def query_rag(query, vector_dbs, top_k):
    if not vector_dbs:
        st.sidebar.error("RAG system is not properly set up. Please check your configuration and try again.")
        return None

    # Merge all vector databases
    merged_db = None
    for file, vector_db in vector_dbs.items():
        if merged_db is None:
            merged_db = vector_db
        else:
            merged_db.merge_from(vector_db)

    if merged_db is None:
        st.sidebar.error("Failed to merge vector databases.")
        return None

    # Perform similarity search on the merged database
    results = merged_db.similarity_search_with_score(query, k=top_k)
    results.sort(key=lambda x: x[1])
    
    contents = " ".join([doc.page_content for doc, score in results])

    # Limit the content to 4000 characters if it exceeds that length
    if len(contents) > 4000:
        contents = contents[:4000]

    return contents

def main():
    st.set_page_config(page_title="Deep Crawl", page_icon="🤖", layout="wide")
    st.title("📚 Deep Crawl Assistant")

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    if 'api_key' not in st.session_state:
        st.session_state.api_key = st.sidebar.text_input("OpenAI API Key")
        if st.sidebar.button("Save OpenAI API Key"):
            open('.env', 'a').write(
                f'OPENAI_API_KEY={st.session_state.api_key}')

    urls = st.sidebar.text_area("URLs to scrape (one per line)", height=100)

    # reading the settings
    with open("settings.json", "r") as settings_file:
        settings = json.load(settings_file)

    model = settings["model"]
    top_k = settings["top_k"]
    chunk_size = settings["chunk_size"]
    chunk_overlap = settings["chunk_overlap"]
    min_content_length = settings["min_content_length"]
    max_depth = settings["max_depth"]

    parent_dir = os.path.abspath(os.path.join(os.getcwd()))
    files = os.listdir(parent_dir)
    st.session_state.vector_dbs = setup_rag(chunk_size, chunk_overlap, files)
    if st.session_state.vector_dbs:
        st.sidebar.success(f"RAG system is ready!")

    if st.sidebar.button("Scrape and Add to Knowledge Base"):
        if not st.session_state.api_key:
            st.sidebar.error("Please enter your OpenAI API Key")
        elif not urls:
            st.sidebar.error("Please enter at least one URL")
        else:
            url_list = urls.split('\n')

            with st.spinner("Scraping websites in parallel..."):
                scraped_urls = scrape_urls_parallel(
                    url_list, max_depth, min_content_length)

            st.sidebar.success(f"Scraped {len(scraped_urls)} URLs successfully")

            with st.spinner("Setting up RAG system..."):
                new_vdbs = setup_rag(chunk_size, chunk_overlap, files)
                st.session_state.vector_dbs.update(new_vdbs)
                print(st.session_state.vector_dbs)

            if st.session_state.vector_dbs:
                st.sidebar.success("RAG system is ready!")
            else:
                st.rerun()

    if st.sidebar.button("Refresh"):
        st.rerun()
    # Main area for querying
    client = OpenAI(api_key=st.session_state.api_key)

    # Define the system prompt
    system_prompt = open('system_prompt.txt', 'r', encoding='utf-8').read()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": system_prompt}
        ]

    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        if 'vector_dbs' not in st.session_state or not st.session_state.vector_dbs:
            st.error("Please scrape and setup the RAG system first.")
        else:
            st.session_state.messages.append(
                {"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            context = query_rag(prompt, st.session_state.vector_dbs, top_k)

            with st.chat_message("assistant"):
                stream = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": m["role"], "content": f"Answer the query '{m['content']}' based on the following contents:\n{context}"}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                )
                response = st.write_stream(stream)
            st.session_state.messages.append(
                {"role": "assistant", "content": response})

            st.markdown(
                f"""<details><summary>Source</summary><p>{context}</p></details> """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
