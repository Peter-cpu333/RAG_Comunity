# 小区推荐系统RAG
"""运行方法
source comm_rag/bin/activate
streamlit run run.py
"""

import os
import bs4
import json
import streamlit as st
# 导入LLM
from langchain_openai import ChatOpenAI
# 导入Embedding模型
from langchain_community.embeddings import DashScopeEmbeddings
# 构建向量库
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores import FAISS
# Prompt
from langchain import hub
# 
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict






@st.cache_resource
def build_rag_graph():
    model = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",  # 此处以qwen-plus为例，您可按需更换模型名称
    )

    embeddings = DashScopeEmbeddings(model = "text-embedding-v3")


    def load_jsonl_as_docs(jsonl_path):
        docs = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # 拼接文本
                text = (
                    f"小区名称：{data.get('小区名称', '')}；"
                    f"小区单价：{data.get('小区均价', '')}；"
                    f"总价：{data.get('价格', '')}；"
                    f"区域：{data.get('区域', '')}；"
                    f"商圈：{data.get('商圈', '')}；"
                    f"其他描述：{data.get('小区解读_其他描述', '')}；"
                    f"小区不足：{data.get('小区解读_小区不足', '')}；"
                    f"小区户型：{data.get('小区解读_小区户型', '')}；"
                    f"小区设施：{data.get('小区解读_小区设施', '')}；"
                    f"居民素质：{data.get('小区解读_居民素质', '')}；"
                    f"房屋品质：{data.get('小区解读_房屋品质', '')}；"
                    f"生活配套：{data.get('小区解读_生活配套', '')}；"
                    f"轨道交通：{data.get('小区解读_轨道交通', '')}\n"
                )
                doc = Document(page_content=text)
                docs.append(doc)
        return docs


    # 用法
    # docs = load_jsonl_as_docs('/Users/zhuyq0719/Documents/文稿 - Zhuyq0719的MacBook Air/小区推荐系统RAG/data/processed_data.jsonl')
    # db = FAISS.from_documents(docs, embeddings)
    # index_path = './faiss_index'
    # # 加载索引
    db = FAISS.load_local('data/faiss_index', embeddings,
    allow_dangerous_deserialization=True)
    # db = FAISS.load_local(index_path, embeddings)
    

    
    # 4.4 构建RAG的Prompt模版
    prompt_template = """请结合下方“小区资料”，根据用户的问题，推荐最符合条件的小区。推荐数量可以根据具体情况灵活调整，通常为3到4个，但不局限于此。请对每个推荐的小区进行详细分析，并说明推荐理由。
    

    除了符合条件的小区外，如果有其他区域的小区虽然不完全符合主要条件但有一定优势（如距离不远、性价比高等），可以作为备选推荐，并详细说明其优缺点。

    请按照以下格式输出：
    
    - 推荐小区（推荐理由，不要输出括号内的内容）
    - 备选小区（推荐理由，说明其不足或不完全符合条件的地方，不要输出括号内的内容）

    注意事项：
    - 当用户仅提问某个小区好坏（即用户已有中意小区时，不要给其推荐备选小区）
    - 对每个推荐小区进行详细的分析，包括性价比、交通便利性、居住舒适度及周边配套设施。
    - 提供对比分析，说明为什么这些小区比其他选择更适合用户需求。
    - 在推荐理由中加入个人化建议，结合用户问题进行深度讲解。
    - 对于备选小区，特别说明其相对优势以及为何值得考虑。

    小区资料：
    {context}

    用户问题：
    {query}
    """

    
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
        retrieved_docs = db.similarity_search(state["question"])
        return {"context": retrieved_docs}


    # 4.5.3
    # 将检索到的文档内容拼接，和问题一起传递给 prompt，生成用于大模型的输入。
    # model.invoke(messages) 调用大模型（如 OpenAI GPT、Llama 等）生成答案。
    # 返回生成的答案。
    from langchain_core.messages import HumanMessage

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        formatted_prompt = prompt_template.format(query=state["question"], context=docs_content)
        
        # 如果 formatted_prompt 是一个字符串，直接传递
        # response = model.invoke(formatted_prompt)
        
        # 或者将字符串转换为 HumanMessage
        message = HumanMessage(content=formatted_prompt)
        
        # 传递消息列表
        response = model.invoke([message])
        
        return {"answer": response.content}
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    return graph

graph = build_rag_graph()


import streamlit as st
from langchain_community.document_loaders import PyPDFLoader

# uploaded_file = st.file_uploader("请上传PDF文件", type=["pdf"])

# if uploaded_file is not None:
#     with open("temp.pdf", "wb") as f:
#         f.write(uploaded_file.read())
#     loader = PyPDFLoader("temp.pdf")
#     docs = loader.load()

st.title("小区推荐系统")

user_input = st.text_input("请输入你的问题：")

if st.button("提交") and user_input:
    with st.spinner("正在生成答案..."):
        response = graph.invoke({"question": user_input})
        st.write("答案：")
        st.write(response["answer"])
