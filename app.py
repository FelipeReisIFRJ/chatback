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

# Prompts personalizados
prompt_pt = """
Você é um assistente virtual altamente preciso e confiável. Sua tarefa é responder perguntas baseadas exclusivamente no seguinte contexto extraído de um documento em PDF.
Responda em português. 
Se a pergunta não puder ser respondida com base no contexto abaixo, diga:
"Não sei responder à pergunta com base no contexto disponível."
Não faça suposições ou invente informações.

Contexto: {context}

Pergunta: {question}

Resposta:
"""

prompt_en = """
You are a highly accurate and reliable virtual assistant. Your task is to answer questions exclusively based on the following context extracted from a PDF document.
Respond in English. 
If the question cannot be answered based on the context below, say:
"I cannot answer the question based on the available context."
Do not make assumptions or invent information.

Context: {context}

Question: {question}

Answer:
"""

# Configurar a interface do Streamlit
st.title("ChatBack")
st.write("Bem-vindo! Conheça as principais diretrizes sobre a dor lombar.")

# Seleção do idioma
lang = st.radio("Selecione o idioma:", ["Português", "English"])

# Selecionar o prompt com base no idioma escolhido
if lang == "Português":
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_pt)
elif lang == "English":
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_en)
else:
    prompt = None

# Campo de texto para entrada do usuário
query = st.text_input("Digite sua pergunta:")

if query and prompt:
    # Criar a cadeia de perguntas e respostas com o prompt selecionado
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )
    result = qa_chain({"query": query})
    st.markdown(f"**Pergunta:** {query}")
    st.markdown(f"**Resposta:** {result['result']}")
