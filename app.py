import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Configurar chave da API da OpenAI
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI"]["API_KEY"]

# Carregar o PDF
@st.cache_resource
def load_vectorstore():
    loader = PyPDFLoader("backpainseries.pdf")
    pages = loader.load_and_split()

    # Dividir texto em segmentos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(pages)

    # Criar embeddings e indexar os documentos
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

vectorstore = load_vectorstore()

# Configurar o modelo LLM
llm = OpenAI(temperature=0)

# Configurar o prompt personalizado
prompt_template = """
Você é um assistente virtual especializado em fornecer respostas baseadas em recomendações atuais sobre dor lombar.
Use apenas as informações fornecidas no seguinte contexto para responder de forma objetiva e sucinta em português.
Se não houver informações suficientes no contexto para responder à pergunta, diga "Não sei responder à pergunta com base no contexto disponível."

Contexto: {context}

Pergunta: {question}

Resposta:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

# Configurar a cadeia de perguntas e respostas com recuperação
chatbot_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=False
)

# Interface do Streamlit
st.title("ChatBack: Tire suas dúvidas sobre Dor Lombar")

if "history" not in st.session_state:
    st.session_state["history"] = []

# Entrada do usuário
user_input = st.text_input("Digite sua pergunta:")

if st.button("Enviar"):
    if user_input:
        result = chatbot_chain({"query": user_input})
        answer = result["result"]
        st.session_state["history"].append({"Pergunta": user_input, "Resposta": answer})

# Exibir histórico
if st.session_state["history"]:
    for i, chat in enumerate(st.session_state["history"]):
        st.write(f"**Pergunta {i+1}:** {chat['Pergunta']}")
        st.write(f"**Resposta:** {chat['Resposta']}")
