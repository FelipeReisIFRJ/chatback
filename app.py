import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import streamlit as st

# Configurar chave da API da OpenAI via secrets do Streamlit
openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key

# Carregar o PDF
@st.cache_resource
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return pages

# Processar o PDF e criar vetorização
@st.cache_resource
def process_documents(_pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    documents = text_splitter.split_documents(_pages)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# Inicializar o vetor de recuperação
file_path = "backpainguide.pdf"  # Nome do arquivo no mesmo diretório
pages = load_pdf(file_path)
vectorstore = process_documents(pages)

# Configurar LLM
llm = OpenAI(temperature=0)

# Configurar o prompt personalizado
prompt_template = """
Você é um assistente virtual altamente preciso e confiável. Sua tarefa é responder perguntas baseadas exclusivamente no seguinte contexto extraído de um documento em PDF.
Se a pergunta não puder ser respondida com base no contexto abaixo, diga:
"Não sei responder à pergunta com base no contexto disponível."
Não faça suposições ou invente informações.

Contexto: {context}

Pergunta: {question}

Resposta:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

# Criar a cadeia de perguntas e respostas
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=False  # Ajustável
)

# Configurar a interface do Streamlit
st.title("ChatBack")
st.write("Olá! Conheça as principais diretrizes sobre a dor lombar.")

# Campo de texto para entrada do usuário
query = st.text_input("Digite sua pergunta:")

if query:
    result = qa_chain({"query": query})
    st.markdown(f"**Pergunta:** {query}")
    st.markdown(f"**Resposta:** {result['result']}")
