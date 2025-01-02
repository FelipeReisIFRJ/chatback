# -*- coding: utf-8 -*-
import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Configurar chave da API da OpenAI
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI"]["API_KEY"]

# Função para carregar o PDF
def load_pdf():
    loader = PyPDFLoader("backpainseries.pdf")
    pages = loader.load_and_split()
    return pages

@st.cache_resource
def setup_vectorstore(pages_content):
    # Dividir texto em segmentos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_text(pages_content)

    # Criar embeddings e indexar os documentos
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(documents, embeddings)
    return vectorstore

# Carregar o PDF
pages = load_pdf()

# Concatenar o conteúdo das páginas em uma string para evitar erro de hash
pages_content = " ".join([page.page_content for page in pages])

# Configurar o vetor de busca
vectorstore = setup_vectorstore(pages_content)

# Configurar o modelo LLM e a Cadeia de Resposta
llm = OpenAI(temperature=0)
chatbot_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=False
)

# Configurar o Streamlit
st.title("ChatBack: Informações sobre Dor Lombar")

# Interface do Chatbot
if "history" not in st.session_state:
    st.session_state["history"] = []

user_input = st.text_input("Digite sua pergunta:")
if st.button("Enviar"):
    if user_input:
        result = chatbot_chain({"query": user_input})
        answer = result["result"]
        st.session_state["history"].append({"Pergunta": user_input, "Resposta": answer})

# Exibir Histórico
if st.session_state["history"]:
    for i, chat in enumerate(st.session_state["history"]):
        st.write(f"**Pergunta {i+1}:** {chat['Pergunta']}")
        st.write(f"**Resposta:** {chat['Resposta']}")

