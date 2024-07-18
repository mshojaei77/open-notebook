import streamlit as st
import json

settings = {
    "model": st.selectbox("OpenAI Model", ["gpt-3.5-turbo", "gpt-4o"]),
    "top_k": st.slider("Number of similar documents to retrieve", 1, 5, 1),
    "chunk_size": st.slider("Chunk size", 500, 4000, 1500),
    "chunk_overlap": st.slider("Chunk overlap", 0, 200, 50),
    "min_content_length": st.slider("Min content length of html tag", 50, 500, 100),
    "max_depth": st.slider("Max crawling depth (more depth take much longer time)", 1, 10, 1)
}

if st.button("Save settings"):
    with open("settings.json", "w") as settings_file:
        json.dump(settings, settings_file)
