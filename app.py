import streamlit as st
import json
import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from crawler import scrape_urls_parallel
from dotenv import load_dotenv
import shutil
import PyPDF2
import uuid
import re

# Define directories for knowledge base
KBASE_DIR = Path("knowledge_base")
JSON_DIR = KBASE_DIR / "json"
FAISS_DIR = KBASE_DIR / "faiss"

# Create directories if they don't exist
JSON_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name):
    # Remove invalid characters for filenames
    return re.sub(r'[^a-zA-Z0-9-_\.]', '_', name)

def generate_readable_filename(base_name):
    # Truncate the base name to a reasonable length for readability
    return '_'.join(base_name.split()[:5]).lower()

def setup_rag(chunk_size, chunk_overlap, files):
    if 'api_key' not in st.session_state or not st.session_state.api_key:
        st.sidebar.error("Please enter your OpenAI API Key in the sidebar.")
        return None

    try:
        embeddings = OpenAIEmbeddings(api_key=st.session_state.api_key)
    except Exception as e:
        st.sidebar.error(f"Error initializing OpenAI Embeddings: {str(e)}")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n'], chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    vector_dbs = {}

    for file in files:
        if file.endswith(".json") and file != "settings.json":
            try:
                file_path = JSON_DIR / file
                embedding_file = FAISS_DIR / file.replace(".json", ".faiss")

                if embedding_file.exists():
                    vector_db = FAISS.load_local(
                        str(embedding_file), embeddings, allow_dangerous_deserialization=True)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError as json_error:
                            st.sidebar.error(f"Error processing file {file}: {str(json_error)}")
                            continue
                    if isinstance(data, dict):
                        text = data.get('content', '') or data.get('pasted_text', '')
                        title = data.get('filename', data.get('title', 'Untitled'))
                    elif isinstance(data, list):
                        text = ' '.join([item.get('content', '') or item.get('pasted_text', '') for item in data])
                        title = "Multiple Documents"
                    else:
                        raise ValueError(f"Unexpected data format in {file}")
                    documents = text_splitter.create_documents([text])
                    vector_db = FAISS.from_documents(documents, embeddings)
                    vector_db.save_local(str(embedding_file))

                vector_dbs[file] = vector_db
            except json.JSONDecodeError as json_error:
                st.sidebar.error(f"Error processing file {file}: {str(json_error)}")
                continue
            except Exception as e:
                st.sidebar.error(f"Error processing file {file}: {str(e)}")
                continue

    return vector_dbs

def query_rag(query, vector_dbs, top_k):
    if not vector_dbs:
        st.sidebar.error("RAG system is not properly set up. Please check your configuration and try again.")
        return None

    merged_db = None
    for file, vector_db in vector_dbs.items():
        if merged_db is None:
            merged_db = vector_db
        else:
            merged_db.merge_from(vector_db)

    if merged_db is None:
        st.sidebar.error("Failed to merge vector databases.")
        return None

    results = merged_db.similarity_search_with_score(query, k=top_k)
    results.sort(key=lambda x: x[1])
    
    contents = " ".join([doc.page_content for doc, score in results])

    if len(contents) > 4000:
        contents = contents[:4000]

    return contents

st.set_page_config(page_title="Open Notebook", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
# Custom CSS for dark theme and improved UI
st.markdown("""
<style>
    /* Global styles */
    body {
        color: #E0E0E0;
        background-color: #1E1E1E;
    }
    
    /* Sidebar styles */
    .css-1d391kg {
        background-color: #252526;
    }
    
    /* Main content area styles */
    .stApp {
        background-color: #1E1E1E;
    }
    
    /* Button styles */
    .stButton>button {
        color: #FFFFFF;
        background-color: #007ACC;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #005A9E;
    }
    
    /* Input field styles */
    .stTextInput>div>div>input, .stTextArea textarea {
        color: #D4D4D4;
        background-color: #3C3C3C;
        border: 1px solid #3C3C3C;
        border-radius: 4px;
    }
    
    /* Selectbox styles */
    .stSelectbox>div>div>select {
        color: #D4D4D4;
        background-color: #3C3C3C;
        border: 1px solid #3C3C3C;
        border-radius: 4px;
    }
    
    /* Slider styles */
    .stSlider>div>div>div>div {
        background-color: #007ACC;
    }
    
    /* Expander styles */
    .stExpander {
        background-color: #252526;
        border: 1px solid #3C3C3C;
        border-radius: 4px;
    }
    .stExpander>summary {
        color: #E0E0E0;
        font-weight: 500;
    }
    
    /* Chat message styles */
    .stChatMessage {
        background-color: #252526;
        border-radius: 4px;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Code block styles */
    pre {
        background-color: #1E1E1E;
        border: 1px solid #3C3C3C;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Load settings
if not (KBASE_DIR / "settings.json").exists():
    default_settings = {
        "model": "gpt-3.5-turbo",
        "top_k": 3,
        "chunk_size": 1500,
        "chunk_overlap": 50,
    }
    with open(KBASE_DIR / "settings.json", "w") as settings_file:
        json.dump(default_settings, settings_file)

with open(KBASE_DIR / "settings.json", "r") as settings_file:
    settings = json.load(settings_file)

model = settings["model"]
top_k = settings["top_k"]
chunk_size = settings["chunk_size"]
chunk_overlap = settings["chunk_overlap"]

text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n'], chunk_size=chunk_size, chunk_overlap=chunk_overlap)

with st.sidebar:
    st.header("Configuration")
    
    # API Key Configuration
    with st.expander("API Key", expanded=False):
        load_dotenv()
        local_api_key = os.getenv('OPENAI_API_KEY')
        if local_api_key:
            st.session_state.api_key = local_api_key
        else:
            st.session_state.api_key = st.text_input(
                "OpenAI API Key", type="password", help="Enter your OpenAI API key here. You can get one from https://platform.openai.com/account/api-keys")
        
        # Add option to change API key
        new_api_key = st.text_input("New OpenAI API Key", type="password", help="Enter a new OpenAI API key if you want to change it")
        if st.button("Update API Key"):
            if new_api_key:
                st.session_state.api_key = new_api_key
                with open('.env', 'w') as env_file:
                    env_file.write(f'OPENAI_API_KEY={new_api_key}')
                st.success("API Key updated successfully!")
            else:
                st.warning("Please enter a new API key to update")

    # Settings
    with st.expander("Advanced Settings", expanded=False):
        #don't change model names "gpt-4o-mini", "gpt-4o" are 2024 new models
        settings = {
            "model": st.selectbox("AI Model", ["gpt-4o-mini", "gpt-4o"], index=1 if settings["model"] == "gpt-4o" else 0, help="Choose the AI model to use. GPT-4 is more capable but slower and more expensive."),
            "top_k": st.slider("Number of relevant documents", 1, 10, settings["top_k"], help="Number of most relevant documents to retrieve for each query. Higher values may improve accuracy but increase processing time."),
            "chunk_size": st.number_input("Chunk Size", min_value=500, max_value=5000, value=settings["chunk_size"], step=100, help="Size of each text chunk for processing."),
            "chunk_overlap": st.number_input("Chunk Overlap", min_value=0, max_value=500, value=settings["chunk_overlap"], step=10, help="Number of overlapping characters between chunks."),
        }

        if st.button("Save settings"):
            with open(KBASE_DIR / "settings.json", "w") as settings_file:
                json.dump(settings, settings_file)
            st.success("Settings saved successfully!")

    # URL Scraping
    with st.expander("Add Websites to Knowledge Base", expanded=False):
        urls = st.text_area("Enter website URLs (one per line)", height=100, help="Enter the URLs of websites you want to add to your knowledge base. The AI will scrape and learn from these websites.")
        if st.button("Add Websites to Knowledge Base"):
            if not st.session_state.api_key:
                st.error("Please enter your OpenAI API Key first")
            elif not urls:
                st.error("Please enter at least one URL")
            else:
                url_list = [url.strip() for url in urls.split('\n') if url.strip()]

                with st.spinner("Reading websites..."):
                    scraped_urls = scrape_urls_parallel(url_list, max_depth=2, min_content_length=100)

                st.success(f"Successfully read {len(scraped_urls)} websites")

                with st.spinner("Updating knowledge base..."):
                    new_files = []
                    for url, content in scraped_urls.items():
                        # Use domain name as the base for the title
                        domain = re.sub(r'^https?://', '', url).split('/')[0]
                        title = content.get("title", domain)
                        sanitized_title = sanitize_filename(title)
                        readable_title = generate_readable_filename(title)
                        unique_id = uuid.uuid4().hex[:6]
                        json_filename = f"{readable_title}_{unique_id}.json"
                        json_path = JSON_DIR / json_filename
                        with open(json_path, "w", encoding='utf-8') as jf:
                            json.dump({"url": url, "title": title, "content": content.get("content", "")}, jf, ensure_ascii=False)
                        new_files.append(json_filename)
                    
                    files = [f.name for f in JSON_DIR.glob("*.json") if f.name != "settings.json"]
                    new_vdbs = setup_rag(chunk_size, chunk_overlap, files)
                    if new_vdbs:
                        st.session_state.vector_dbs = new_vdbs
                        st.success("Knowledge base updated successfully!")
                    else:
                        st.error("Failed to update knowledge base. Please try again.")

                st.rerun()

    # Add PDF Upload
    with st.expander("Add PDFs to Knowledge Base", expanded=False):
        uploaded_pdfs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True, help="Upload PDF files to add to your knowledge base. The AI will read and learn from these documents.")
        if st.button("Add PDFs to Knowledge Base"):
            if not uploaded_pdfs:
                st.error("Please upload at least one PDF file.")
            else:
                for pdf in uploaded_pdfs:
                    try:
                        pdf_reader = PyPDF2.PdfReader(pdf)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text() or ""
                        
                        if len(text) < 100:
                            st.warning(f"PDF '{pdf.name}' content is too short and was skipped.")
                            continue

                        sanitized_name = sanitize_filename(pdf.name.replace('.pdf', ''))
                        readable_name = generate_readable_filename(sanitized_name)
                        unique_id = uuid.uuid4().hex[:6]
                        json_filename = f"{readable_name}_{unique_id}.json"
                        json_path = JSON_DIR / json_filename
                        with open(json_path, "w", encoding='utf-8') as jf:
                            json.dump({
                                "filename": pdf.name,
                                "title": sanitized_name,
                                "content": text
                            }, jf, ensure_ascii=False)

                    except Exception as e:
                        st.error(f"Failed to process PDF '{pdf.name}': {str(e)}")
                
                with st.spinner("Updating knowledge base..."):
                    files = [f.name for f in JSON_DIR.glob("*.json") if f.name != "settings.json"]
                    new_vdbs = setup_rag(chunk_size, chunk_overlap, files)
                    if new_vdbs:
                        st.session_state.vector_dbs = new_vdbs
                        st.success("PDFs added to knowledge base successfully!")
                    else:
                        st.error("Failed to update knowledge base. Please try again.")
                
                st.rerun()

    # Add Text Input
    with st.expander("Add Custom Text to Knowledge Base", expanded=False):
        pasted_text = st.text_area("Enter or paste your text here:", height=200, help="Enter or paste any custom text you want to add to your knowledge base.")
        custom_title = st.text_input("Title for the custom text", help="Provide a title to easily identify this custom text.")
        if st.button("Add Text to Knowledge Base"):
            if not pasted_text.strip():
                st.error("Please enter some text to add.")
            elif len(pasted_text.strip()) < 100:
                st.error("The text is too short. Please enter at least 100 characters.")
            elif not custom_title.strip():
                st.error("Please provide a title for the custom text.")
            else:
                try:
                    # Use the first five words of the text if title is not sufficiently descriptive
                    if len(custom_title.split()) < 3:
                        first_five = ' '.join(pasted_text.strip().split()[:5])
                        custom_title = f"{custom_title} - {first_five}"
                    
                    sanitized_title = sanitize_filename(custom_title)
                    readable_title = generate_readable_filename(custom_title)
                    unique_id = uuid.uuid4().hex[:6]
                    json_filename = f"{readable_title}_{unique_id}.json"
                    json_path = JSON_DIR / json_filename
                    with open(json_path, "w", encoding='utf-8') as jf:
                        json.dump({
                            "title": custom_title,
                            "pasted_text": pasted_text
                        }, jf, ensure_ascii=False)

                    with st.spinner("Updating knowledge base..."):
                        files = [f.name for f in JSON_DIR.glob("*.json") if f.name != "settings.json"]
                        new_vdbs = setup_rag(chunk_size, chunk_overlap, files)
                        if new_vdbs:
                            st.session_state.vector_dbs = new_vdbs
                            st.success("Custom text added to knowledge base successfully!")
                        else:
                            st.error("Failed to update knowledge base. Please try again.")
                    
                    st.rerun()

                except Exception as e:
                    st.error(f"Failed to add custom text: {str(e)}")

    # Knowledge Base Management
    with st.expander("Manage Knowledge Base", expanded=False):
        st.subheader("Current Knowledge Base")
        files = list(JSON_DIR.glob("*.json"))
        
        if "vector_dbs" not in st.session_state:
            st.session_state.vector_dbs = setup_rag(chunk_size, chunk_overlap, [f.name for f in files if f.name != "settings.json"])

        displayed_items = set()

        for json_file in files:
            if json_file.name in displayed_items:
                continue
            try:
                with open(json_file, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                title = data.get('title') or data.get('filename') or "Untitled"
            except Exception:
                title = json_file.stem

            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(title)
            with col2:
                if st.button("Remove", key=f"remove_{json_file.name}"):
                    try:
                        faiss_file = FAISS_DIR / json_file.name.replace(".json", ".faiss")
                        json_file.unlink()
                        if faiss_file.exists():
                            shutil.rmtree(faiss_file)
                        st.success(f"Removed '{title}' from the knowledge base")
                        st.session_state.vector_dbs.pop(json_file.name, None)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error removing '{title}': {str(e)}")
            displayed_items.add(json_file.name)

        if st.button("Clear Entire Knowledge Base"):
            try:
                for json_file in files:
                    json_file.unlink()
                for faiss_dir in FAISS_DIR.glob("*"):
                    if faiss_dir.is_dir():
                        shutil.rmtree(faiss_dir)
                st.session_state.vector_dbs = {}
                st.success("Knowledge base cleared successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing knowledge base: {str(e)}")

    if st.button("Refresh"):
        st.rerun()

client = OpenAI(api_key=st.session_state.api_key)

system_prompt = open('system_prompt.txt', 'r', encoding='utf-8').read()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt}
    ]

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about your knowledge base"):
    if 'vector_dbs' not in st.session_state or not st.session_state.vector_dbs:
        st.error("Your knowledge base is empty. Please add some content first.")
    else:
        st.session_state.messages.append(
            {"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        context = query_rag(prompt, st.session_state.vector_dbs, top_k)

        if context:
            with st.chat_message("assistant"):
                stream = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Answer the query '{prompt}' based on the following contents:\n{context}"}
                    ],
                    stream=True,
                )
                response = st.write_stream(stream)
            st.session_state.messages.append(
                {"role": "assistant", "content": response})

            st.markdown(
                f"""<details><summary>Source Information</summary><p>{context}</p></details>""", unsafe_allow_html=True)
