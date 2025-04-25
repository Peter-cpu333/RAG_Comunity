from langchain_core.messages import AIMessage

import re
def build_rag():

    # 导入LLM
    # 初始化模型
    from langchain_openai import ChatOpenAI
    import os
    from langchain_community.vectorstores import FAISS
    # 1. 导入LLM
    # 初始化模型
    llm = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",  # 此处以qwen-plus为例，您可按需更换模型名称
    )

    # 2. 导入Embedding模型
    from langchain_community.embeddings import DashScopeEmbeddings

    embeddings = DashScopeEmbeddings(model = "text-embedding-v3")

    vector_store = FAISS.load_local('/Users/zhuyq0719/小区推荐系统RAG/data/faiss_index', embeddings,
        allow_dangerous_deserialization=True)
        

    
        
    
    from langchain_core.tools import tool
    from langgraph.graph import MessagesState, StateGraph

    graph_builder = StateGraph(MessagesState)

    from langchain_core.tools import tool


    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve information related to a query."""
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    from langchain_core.messages import SystemMessage
    from langgraph.prebuilt import ToolNode


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
    def query_or_respond(state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])
        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}


    # Step 2: Execute the retrieval.
    tools = ToolNode([retrieve])


    # Step 3: Generate a response using the retrieved content.
    def generate(state: MessagesState):
        """Generate answer."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        
        system_message_content = """请结合下方“小区资料”，根据用户的问题，推荐最符合条件的小区。推荐数量可以根据具体情况灵活调整，通常为3到4个，但不局限于此。请对每个推荐的小区进行详细分析，并说明推荐理由。
        除了符合条件的小区外，如果有其他区域的小区虽然不完全符合主要条件但有一定优势（如距离不远、性价比高等），可以作为备选推荐，并详细说明其优缺点。
        请按照以下格式输出：
        - 推荐小区（推荐理由，不要输出括号内的内容）
        - 备选小区（推荐理由，说明其不足或不完全符合条件的地方，不要输出括号内的内容）

        注意事项：
        - 详细解释第一个小区，之后的可以简略
        - 当用户仅提问某个小区好坏（即用户已有中意小区时，不要给其推荐备选小区）
        - 对每个推荐小区进行详细的分析，包括性价比、交通便利性、居住舒适度及周边配套设施。
        - 提供对比分析，说明为什么这些小区比其他选择更适合用户需求。
        - 在推荐理由中加入个人化建议，结合用户问题进行深度讲解。
        - 对于备选小区，特别说明其相对优势以及为何值得考虑。

        小区资料：
        f"{docs_content}"

        
        """

        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = llm.invoke(prompt)
        return {"messages": [response]}
    
    from langgraph.graph import END
    from langgraph.prebuilt import ToolNode, tools_condition

    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)
    return graph_builder


graph = build_rag().compile()

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
        final_answer = ""
        response_placeholder = st.empty()
        for event in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="values",
            config={"configurable": {"thread_id": "def234"}},
        ):
            if "messages" in event:
                
                messages = event["messages"]
                for message in messages:
                    if isinstance(message, AIMessage):
                        partial_answer = message.content
                        final_answer += partial_answer
                        response_placeholder.markdown(final_answer + "▌")

        response_placeholder.markdown(final_answer)
            

        