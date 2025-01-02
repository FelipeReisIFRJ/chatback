import os 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from openai.exceptions import Timeout, APIError, APIConnectionError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import streamlit as st

# Function to create retry decorator for OpenAI API calls
def _create_retry_decorator(embeddings):
    return retry(
        stop=stop_after_attempt(embeddings.max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(
            retry_if_exception_type(Timeout)
            | retry_if_exception_type(APIError)
            | retry_if_exception_type(APIConnectionError)
            | retry_if_exception_type(RateLimitError)
        ),
    )

# Function to set up the FAISS vector store
@st.cache_resource
def setup_vectorstore(documents):
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(documents, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error setting up vector store: {e}")
        raise

# Load PDF content
def load_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        return [page.page_content for page in pages]
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        raise

# Main Streamlit application
def main():
    st.title("Document Search App with OpenAI and FAISS")

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully!")

        # Load and process PDF
        st.info("Processing PDF...")
        documents = load_pdf("temp.pdf")
        vectorstore = setup_vectorstore(documents)

        # Search functionality
        query = st.text_input("Enter your search query:")
        if query:
            st.info("Searching...")
            try:
                results = vectorstore.similarity_search(query, k=5)
                for i, result in enumerate(results):
                    st.subheader(f"Result {i+1}")
                    st.write(result.page_content)
            except Exception as e:
                st.error(f"Error during search: {e}")

if __name__ == "__main__":
    main()
