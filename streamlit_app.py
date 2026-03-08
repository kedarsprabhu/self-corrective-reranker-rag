import streamlit as st
import httpx
import json
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Self RAG Chat", page_icon="🧠", layout="wide")

API_BASE = "http://127.0.0.1:8000"
CLIENT_ID = os.getenv("CLIENT_ID", "catesigo123")
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "internal164")

# Initialize session state
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "chat_session" not in st.session_state:
    st.session_state.chat_session = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "selected_docs" not in st.session_state:
    st.session_state.selected_docs = []

def authenticate():
    try:
        resp = httpx.post(
            f"{API_BASE}/token",
            json={"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}
        )
        resp.raise_for_status()
        st.session_state.auth_token = resp.json()["access_token"]
    except Exception as e:
        st.error(f"Authentication failed: {e}")

def get_headers():
    return {"Authorization": f"Bearer {st.session_state.auth_token}"}

def load_documents():
    if not st.session_state.auth_token:
        return
    try:
        resp = httpx.get(f"{API_BASE}/v1/documents", headers=get_headers())
        resp.raise_for_status()
        st.session_state.documents = resp.json().get("documents", [])
    except Exception as e:
        st.error(f"Failed to load documents: {e}")

def upload_document(uploaded_file):
    with st.spinner(f"Uploading and processing {uploaded_file.name}..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            resp = httpx.post(
                f"{API_BASE}/v1/process-document", 
                headers=get_headers(), 
                files=files,
                data={"extract_images": "true"}
            )
            if resp.status_code in (200, 206):
                st.success(f"{uploaded_file.name} uploaded successfully!")
                load_documents()
            else:
                st.error(f"Upload failed: {resp.text}")
        except Exception as e:
            st.error(f"Upload error: {e}")

if st.session_state.auth_token is None:
    authenticate()
    if st.session_state.auth_token:
        load_documents()

# Sidebar
with st.sidebar:
    st.title("🧠 Documents")
    
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt", "doc", "docx"], label_visibility="collapsed")
    if uploaded_file is not None:
        if st.button("📄 Process Document", use_container_width=True):
            upload_document(uploaded_file)
            
    st.divider()
    
    if not st.session_state.documents:
        st.info("No documents yet. Upload one to get started.")
    else:
        st.subheader(f"Available Documents ({len(st.session_state.documents)})")
        selected_docs = []
        for doc in st.session_state.documents:
            # Pre-select if it was selected before
            is_checked = doc["file_name"] in st.session_state.selected_docs
            if st.checkbox(f'{doc["object_key"]}', value=is_checked, key=doc["id"]):
                selected_docs.append(doc["file_name"])
        st.session_state.selected_docs = selected_docs

# Main Chat
st.title("Self RAG Chat")

if not st.session_state.selected_docs:
    st.warning("Please select at least one document from the sidebar to start chatting.")

# Display chat messages history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "meta" in msg and msg["meta"]:
            meta = msg["meta"]
            
            if meta.get("confidence_score") is not None:
                st.caption(f"Confidence: {meta['confidence_score']*100:.0f}%")
                
            if meta.get("supporting_facts"):
                with st.expander(f"{len(meta['supporting_facts'])} Supporting Facts"):
                    for fact in meta["supporting_facts"]:
                        st.markdown(f"- {fact}")
                        
            if meta.get("sources"):
                with st.expander(f"{len(meta['sources'])} Source(s)"):
                    for source in meta["sources"]:
                        st.markdown(f"- {source}")

# User Input
if prompt := st.chat_input("Ask a question about your documents...", disabled=not st.session_state.selected_docs):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        payload = {
            "query": prompt,
            "chat_session": st.session_state.chat_session,
            "source": st.session_state.selected_docs
        }
        
        full_response = ""
        meta_data = {}
        
        try:
            with httpx.stream("POST", f"{API_BASE}/v1/chat-completion", json=payload, headers=get_headers(), timeout=120) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                            if event.get("event") == "text":
                                full_response += event["data"]
                                message_placeholder.markdown(full_response + "▌")
                            elif event.get("event") == "final_response":
                                final_data = event["data"]
                                full_response = final_data.get("answer", full_response)
                                meta_data = {
                                    "confidence_score": final_data.get("confidence_score"),
                                    "supporting_facts": final_data.get("supporting_facts", []),
                                    "sources": final_data.get("sources", [])
                                }
                                message_placeholder.markdown(full_response)
                            elif "error" in event:
                                message_placeholder.error(f"Error: {event['error']}")
                                break
                        except json.JSONDecodeError:
                            pass
                            
                # Fallback if final_response wasn't received but we got streamed text
                if full_response and not meta_data:
                     message_placeholder.markdown(full_response)

        except Exception as e:
            message_placeholder.error(f"Failed to get response: {e}")
            
        # Draw meta expanding UI below the completed message
        if meta_data.get("confidence_score") is not None:
            st.caption(f"Confidence: {meta_data['confidence_score']*100:.0f}%")
            
        if meta_data.get("supporting_facts"):
            with st.expander(f"{len(meta_data['supporting_facts'])} Supporting Facts"):
                for fact in meta_data["supporting_facts"]:
                    st.markdown(f"- {fact}")
                    
        if meta_data.get("sources"):
            with st.expander(f"{len(meta_data['sources'])} Source(s)"):
                for source in meta_data["sources"]:
                    st.markdown(f"- {source}")
                    
        # Save state
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "meta": meta_data
        })
