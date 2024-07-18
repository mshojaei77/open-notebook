import os
import json
import streamlit as st
import shutil

parent_dir = os.path.abspath(os.path.join(os.getcwd()))

files = os.listdir(parent_dir)

json_files = [f for f in files if f.endswith(".json") and f != "settings.json"]
faiss_dirs = [d for d in files if os.path.isdir(os.path.join(parent_dir, d)) and d.endswith(".faiss")]

json_contents = []

for json_file in json_files:
    file_path = os.path.join(parent_dir, json_file)
    with open(file_path, 'r') as f:
        data = json.load(f)
        json_contents.append(data)

# Display the JSON contents using Streamlit
st.title("JSON Data Viewer")
for idx, content in enumerate(json_contents):
    st.subheader(f"Contents of {json_files[idx]}")
    st.json(content)

    # Add a button to delete the JSON file
    if st.sidebar.button(f"Delete {json_files[idx]}"):
        try:
            file_to_delete = os.path.join(parent_dir, json_files[idx])
            if os.path.exists(file_to_delete):
                os.remove(file_to_delete)
            st.success(f"Deleted {json_files[idx]}")
        except PermissionError:
            st.error(f"Permission denied: unable to delete {json_files[idx]}")
        except FileNotFoundError:
            st.error(f"File not found: {json_files[idx]}")
        st.rerun()

# Add buttons to delete faiss directories
for faiss_dir in faiss_dirs:
    if st.sidebar.button(f"Delete {faiss_dir}"):
        try:
            dir_to_delete = os.path.join(parent_dir, faiss_dir)
            if os.path.exists(dir_to_delete):
                os.rmdir(dir_to_delete)
            st.success(f"Deleted {faiss_dir}")
        except PermissionError:
            st.error(f"Permission denied: unable to delete {faiss_dir}")
        except FileNotFoundError:
            st.error(f"Directory not found: {faiss_dir}")
        except OSError:
            shutil.rmtree(dir_to_delete, ignore_errors=True)
            st.success(f"Force deleted {faiss_dir} due to OSError")
        st.rerun()

# Add a refresh button to recheck for files and rerun
if st.sidebar.button("Refresh"):
    st.rerun()