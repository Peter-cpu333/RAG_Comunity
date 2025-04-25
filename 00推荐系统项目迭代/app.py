import streamlit as st
# 导入LLM
# 初始化模型
from langchain_openai import ChatOpenAI
import os
# 2. 导入Embedding模型
from langchain_community.embeddings import DashScopeEmbeddings
# 3. 构建向量库
from langchain_core.vectorstores import InMemoryVectorStore

# 4. RAG链
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# 4.1.2 PDF
from langchain_community.document_loaders import PyPDFLoader



@st.cache_resource
def build_rag_graph():
    model = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",  # 此处以qwen-plus为例，您可按需更换模型名称
    )

    embeddings = DashScopeEmbeddings(model = "text-embedding-v3")

    vector_store = InMemoryVectorStore(embeddings)

    # 加载PDF文档
    loader = PyPDFLoader("./data/test.pdf")
    docs = loader.load()
    # 4.2 文本切割器和分割文档
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_spliter.split_documents(docs)
    # 4.3 构建索引
    _ = vector_store.add_documents(documents=all_splits)
    

    
    # 4.4 构建RAG的Prompt模版
    prompt = hub.pull("rlm/rag-prompt")
    # 4.5 
    # Define state for application
    # 4.5.1 State 用于存储应用当前的状态，包括用户问题、检索到的上下文文档和最终答案。
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    # 4.5.2 
    # 这个函数接收当前状态（含用户问题），调用 vector_store.similarity_search 检索相关文档，返回新的 context。
    # vector_store 是你的向量数据库对象，比如 FAISS、Milvus、Chroma 等。
    # Define application steps
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}


    # 4.5.3
    # 将检索到的文档内容拼接，和问题一起传递给 prompt，生成用于大模型的输入。
    # model.invoke(messages) 调用大模型（如 OpenAI GPT、Llama 等）生成答案。
    # 返回生成的答案。
    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = model.invoke(messages)
        return {"answer": response.content}
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    return graph

graph = build_rag_graph()


import streamlit as st
from langchain_community.document_loaders import PyPDFLoader

uploaded_file = st.file_uploader("请上传PDF文件", type=["pdf"])

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

st.title("PDF/网页文档智能问答")

user_input = st.text_input("请输入你的问题：")

if st.button("提交") and user_input:
    with st.spinner("正在生成答案..."):
        response = graph.invoke({"question": user_input})
        st.write("答案：")
        st.write(response["answer"])
