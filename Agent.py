"""
Agent-RAG
Author: Zhuyq
Date: 2025/05/01
此Agent的功能包括：
1. 识别用户输入的意图
2. 提取用户查询的小区名称
3. 检索特定小区的信息
4. 根据检索到的信息生成针对用户查询的回答
5. 根据用户的偏好推荐一个合适的小区
"""

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated, List, TypedDict, Dict, Literal, Any
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

class RecommendationState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]  # 聊天消息历史，使用add_messages会自动追加
    user_preferences: Dict[str, Any]  # <--- 替换为大写的 Any
    queried_apartment: str  # 用户查询的特定小区名称 (如果用户有指定)
    # 根据您最终的代码，这个字段的类型应与 recommend_apartment 返回的结构一致，
    # 如果 recommend_apartment 返回 {'candidate_apartments': List[Document], ...}
    # 那么这里的类型可以是 Dict[str, Any] 或者更精确的 Dict[str, List[Document]]
    # 保持 Dict[str, Any] 可以消除警告，但 Dict[str, List[Document]] 更准确如果结构固定
    # 我们先用 Dict[str, Any] 来解决 any 的问题
    candidate_apartments: Dict[str, Any]  # <--- 替换为大写的 Any (或者更精确的 Dict[str, List[Document]])
    apartment_info: Dict[str, Any]  # <--- 替换为大写的 Any
    recommendation_reasoning: str  # 推荐的理由
    conversation_history: List[str] # 记录对话历史 (可选，messages 可能已包含)
    search_results: List[Document] # 网络搜索结果 (如果需要)
    intent: Literal["查询特定小区信息", "请求小区推荐", "其他", "欢迎", "感谢"] # 用户当前的意图


llm = ChatTongyi(
    model_name="deepseek-r1",
    temperature=0,
)

intent_system = (
"""
你的任务是判断用户输入的意图。可能的意图包括：查询特定小区信息，请求小区推荐，以及其他。请根据用户输入返回最合适的意图标签。

只输出以下情况，
查询特定小区信息

请求小区推荐

其他
不要输出其他文本，只要输出意图标签。
“”
"""
)
intent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", intent_system),
        ("human", "{input}"),
    ]
)

def identify_user_intent(state: RecommendationState):
    """
    Langgraph 节点：使用 LLM 判断用户输入的意图并更新 GraphState。

    Args:
        state: 当前的 RecommendationState。

    Returns:
        包含更新后的 intent 的字典。
    """
    last_user_message = state["messages"][-1].content
    intent_chain = intent_prompt | llm
    intent = intent_chain.invoke({"input": last_user_message}).content
    return {"intent": intent}

embeddings = DashScopeEmbeddings(model = "text-embedding-v3")
vectorstores = FAISS.load_local('/Users/zhuyq0719/00我的Agent/01第一个Agent/RAG数据/faiss_index', embeddings, allow_dangerous_deserialization=True)
retriever = vectorstores.as_retriever()



# 定义提取小区名称的提示和链 (chain)
extract_apartment_name_system_prompt_text = """
你的任务是从用户输入中准确提取出提及的房产小区名称。
用户可能会说“我想查一下[小区名称]的信息”，或者“[小区名称]这个小区怎么样？”等等。
请只返回提取到的小区名称。如果用户输入中没有明确提及小区名称，或者无法判断，请返回字符串 "提取失败"。

例如：
用户输入: "我想了解一下万科城市花园这个小区怎么样？"
你的输出: "万科城市花园"

用户输入: "北京有什么好小区推荐吗？"
你的输出: "提取失败"

用户输入: "帮我查查阳光一百。"
你的输出: "阳光一百"

用户输入: "未来之家小区有房吗"
你的输出: "未来之家"
"""
extract_apartment_name_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", extract_apartment_name_system_prompt_text),
        ("human", "请从以下用户输入中提取小区名称：\n---\n{user_input}\n---"),
    ]
)

# 从用户输入中提取小区名称
def extract_apartment_name(state: RecommendationState):
    """
    Langgraph 节点：从用户输入中提取小区名称。
    这里简化处理，直接使用用户消息内容作为小区名称。
    实际应用可能需要更复杂的解析来精确提取小区名称。
    """
    print("---提取小区名称---")
    if not state["messages"]:
        print("错误：消息列表为空，无法提取小区名称。")
        return {"queried_apartment": "提取失败"}
    last_user_message_content = ""
    # 从后往前找最新的用户消息
    for msg in reversed(state["messages"]):
        if msg.type == "human": # Langchain core messages use .type
            last_user_message_content = msg.content
            break
    
    if not last_user_message_content:
        print("错误：未找到用户消息，无法提取小区名称。")
        return {"queried_apartment": "提取失败"}

    print(f"接收到用户输入进行小区名称提取: {last_user_message_content}")
    extract_chain = extract_apartment_name_prompt | llm | StrOutputParser()
    try:
        extracted_name = extract_chain.invoke({"user_input": last_user_message_content})
        extracted_name = extracted_name.strip() # 去除可能的首尾空格
    except Exception as e:
        print(f"调用LLM提取小区名称时发生错误: {e}")
        extracted_name = "提取失败"

    if extracted_name and extracted_name != "提取失败" and extracted_name != '"提取失败"': # LLM有时可能返回带引号的 "提取失败"
        print(f"成功提取到小区名称: '{extracted_name}'")
        return {"queried_apartment": extracted_name}
    else:
        print(f"未能从用户输入 '{last_user_message_content}' 中提取到明确的小区名称。LLM返回: '{extracted_name}'")
        return {"queried_apartment": "提取失败"} # 使用特定标记表示失败



def retriever_specific_apartment_info(state: RecommendationState):
    """
    Langgraph 节点：根据识别出的意图和用户查询的小区名称，检索特定小区的信息。
    假设用户查询的小区名称已经通过某种方式（例如，一个解析节点）提取并存储在 state["queried_apartment"] 中。
    """
    print("\n---开始检索特定小区的信息---")
    # 修正这里的变量名
    query_appartment_name = state.get("queried_apartment") # 使用 .get() 方法获取字段，避免 KeyError
    if not retriever:
        print("错误：Retriever 未初始化。跳过小区信息检索。")
        return {"search_results": []} # 返回空列表
    # 修正这里的变量名，与上面获取值的变量名一致
    if not query_appartment_name or query_appartment_name == "提取失败":
        print(f"错误或未提取到有效小区名称 ('{query_appartment_name}')。跳过小区信息检索。")
        return {"search_results": []} # 返回空列表表示未执行或无结果
    print(f"尝试检索小区: '{query_appartment_name}' 的信息...")
    try:

        search_results = retriever.invoke(query_appartment_name)
        print(f"为小区 '{query_appartment_name}' 检索到 {len(search_results)} 条结果。")
        # 返回更新后的状态，只包含 search_results
        # 在实际 LangGraph 中，节点函数返回的是要更新状态的字典，不是整个状态
        return {"search_results": search_results}

    except Exception as e:
        print(f"检索小区信息时发生错误: {e}")
        return {"search_results": []} # 返回空列表表示检索失败


def spe_format_documents_for_prompt(docs: List[Document]) -> str:
    """
    Formats a list of Document objects into a single string for the LLM prompt.
    """
    if not docs:
        return "没有找到相关的具体信息。"
    # 您可以根据文档的结构和LLM的喜好调整这里的格式
    return "\n\n---\n\n".join([f"相关资料片段 {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

# System prompt for the answer generation LLM
generate_answer_system_prompt_text = """你是一位耐心、专业且乐于助人的房产顾问。
你的任务是根据用户提出的问题以及下面提供的“背景资料”来清晰、准确地回答用户。
- 请直接回答用户的问题。
- 如果背景资料中没有足够的信息来回答某个特定的方面，请明确告知用户“关于[具体方面]，我目前没有找到足够的信息”，而不是编造答案。
- 如果用户的问题与房产无关，请礼貌地说明你主要提供房产相关咨询。
- 回答时请保持友好和专业的语气。
"""

def generate_specific_apartment_answer(state: RecommendationState):
    """
    Langgraph 节点：根据检索到的特定小区信息，生成针对用户查询的回答。
    假设检索到的信息已经存储在 state["search_results"] 中。
    """
    print("\n---开始生成针对特定小区的回答---")
    user_original_question = ""
    # 从后往前找最新的用户消息
    for msg in reversed(state.get("messages", [])):
        if msg.type == "human": # 'human' for HumanMessage
            user_original_question = msg.content
            break
    
    if not user_original_question:
        print("错误：在消息历史中未能找到用户的原始问题。")
        # 通常这种情况不应该发生，如果发生了，返回一个通用的错误消息
        ai_response = AIMessage(content="抱歉，我似乎没有捕捉到您的问题，能麻烦您再说一遍吗？")
        return {"messages": [ai_response]}
    # 找到小区名字以及检索结果
    queried_apartment_name = state.get("queried_apartment") # 可能为 "提取失败" 或实际小区名
    search_results = state.get("search_results", [])
    # 构建LLM的提示
    human_prompt_for_llm = ""

    if queried_apartment_name and queried_apartment_name != "提取失败":
        # 情况1: 成功提取到小区名称
        formatted_docs = spe_format_documents_for_prompt(search_results)
        
        if not search_results: # 提取到小区名，但检索无结果
            human_prompt_for_llm = (
                f"用户咨询关于小区“{queried_apartment_name}”的问题。\n"
                f"用户原始问题是：“{user_original_question}”\n\n"
                f"背景资料：我未能检索到关于“{queried_apartment_name}”的具体信息。\n\n"
                f"请告知用户这一情况（未能找到“{queried_apartment_name}”的详细信息），并询问是否可以帮助了解其他小区或提供其他方面的咨询。"
            )
        else: # 提取到小区名，且有检索结果
            human_prompt_for_llm = (
                f"用户咨询关于小区“{queried_apartment_name}”的问题。\n"
                f"用户原始问题是：“{user_original_question}”\n\n"
                f"背景资料：\n{formatted_docs}\n\n"
                f"请仔细阅读并理解以上背景资料，然后清晰、准确地回答用户的原始问题。"
            )
    elif queried_apartment_name == "提取失败":
        # 情况2: 未能提取到明确的小区名称
        human_prompt_for_llm = (
            f"用户提出了一个问题，但我未能从中准确识别出他们想要查询的具体小区名称。\n"
            f"用户原始问题是：“{user_original_question}”\n\n"
            f"请礼貌地告知用户我没有理解他们想问哪个小区，并请他们提供更明确的小区名称，以便我进行查询。"
        )
    else:
        # 情况3: queried_apartment_name 为 None 或空字符串 (理论上，如果意图正确，前置节点应处理)
        # 这是一个回退情况，可能表示流程中存在意外状态
        print(f"警告：queried_apartment 状态异常 ('{queried_apartment_name}')，将尝试基于原始问题生成通用回复。")
        human_prompt_for_llm = (
            f"用户原始问题是：“{user_original_question}”\n\n"
            f"背景资料：目前没有特定小区的上下文信息。\n\n"
            f"请尝试根据常识回答用户的问题。如果问题明显是关于某个地点但未指明，可以请用户提供更多详细信息。"
        )

    # 构建完整的提示给LLM
    answer_generation_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", generate_answer_system_prompt_text),
            ("human", human_prompt_for_llm),
        ]
    )
    
    # 假设 llm 和 StrOutputParser 已经像您之前那样定义好了
    # llm = ChatTongyi(...)
    # output_parser = StrOutputParser()
    answer_chain = answer_generation_prompt | llm | StrOutputParser()
    
    print(f"提交给LLM生成答案的Human Prompt (部分内容): {human_prompt_for_llm[:400]}...")

    try:
        final_answer_content = answer_chain.invoke({}) # invoke({}) 因为所有变量已在 prompt 中
    except Exception as e:
        print(f"调用LLM生成答案时发生严重错误: {e}")
        final_answer_content = "抱歉，我在为您生成回答时遇到了一些技术问题，请您稍后再试或换个方式提问。"

    ai_response = AIMessage(content=final_answer_content.strip())
    
    # Langgraph 的 add_messages 会自动将这个新消息追加到 state["messages"]
    return {"messages": [ai_response]}


def recommend_apartment(state: RecommendationState):
    """
    Langgraph 节点：根据用户的偏好推荐一个合适的小区。
    假设用户偏好已经存储在 state["user_preferences"] 中。
    """
    print("\n---开始推荐小区---")
    user_original_question = ""
    # 从后往前找最新的用户消息
    for msg in reversed(state.get("messages", [])):
        # *** 修正这里：使用 hasattr(msg, 'type') 和 msg.type ***
        # 同时保留对字典格式的兼容性检查
        if hasattr(msg, 'type') and msg.type == "human":
             user_original_question = msg.content # *** 使用 .content 获取消息内容 ***
             break
        elif isinstance(msg, dict) and msg.get('type') == 'human':
             user_original_question = msg.get('content', '')
             if user_original_question:
                break
    print(f"用户原始问题：{user_original_question}")
    user_preferences = state.get("user_preferences", {})
    if not user_preferences:
        print("未找到用户偏好信息，无法推荐小区。")
        return {"candidate_apartments": [], "recommendation_reasoning": "未提供偏好信息，无法推荐小区"}
    # 假设我们有一个小区数据库或检索器（如 FAISS 或其他数据库）
    try:
        processed_values = []
        for value in user_preferences.values():
            if isinstance(value, list):
                # 如果值是列表，将列表内部元素用逗号或其他分隔符连接
                processed_values.append(",".join(map(str, value))) # 用逗号连接列表元素
            else:
                # 如果不是列表，直接将其转换为字符串
                processed_values.append(str(value))

        query = " ".join(processed_values) # 现在拼接处理过的字符串列表
        query_with_context = f"{query} 用户问题: {user_original_question}"
        print(f"根据以下条件检索推荐小区：{query}")
        recommended_apartments = retriever.invoke(query_with_context)
        if recommended_apartments:
            print(f"成功推荐了 {len(recommended_apartments)} 个小区。")
            reasoning = f"根据您的偏好（{query}），以下是推荐的小区列表："
        else:
            print("未找到符合偏好的小区。")
            reasoning = f"根据您的偏好（{query}），未找到符合的小区。"
        return {
            "candidate_apartments": recommended_apartments,
            "recommendation_reasoning": reasoning,
        }
    except Exception as e:
        print(f"推荐小区时发生错误: {e}")
        return {
            "candidate_apartments": [],
            "recommendation_reasoning": "推荐过程中发生错误，请稍后再试。",
        }


def rec_format_documents_for_prompt(documents: List[Document]) -> str:
    """
    模拟将 LangChain Document 对象列表格式化为字符串。
    从 Document 的 page_content 中提取文本。
    """
    if not documents:
        return ""

    formatted_string = "以下是相关的房源信息：\n\n"
    for i, doc in enumerate(documents):
        # 确保是 Document 对象并有 page_content
        if isinstance(doc, Document) and hasattr(doc, 'page_content'):
            content = doc.page_content.strip()
            if content:
                # 简单地添加文档内容，或者你可以进一步解析 page_content 中的键值对进行更精细的格式化
                formatted_string += f"小区信息 {i+1}:\n{content}\n---\n"
        else:
            print(f"警告: 发现非 Document 对象或缺少 page_content 的项: {doc}")
            # 可以选择跳过或加入提示信息

    return formatted_string.strip()

def generate_recommendation_answer(state: dict) -> dict:
    """
    根据推荐的小区列表和用户偏好生成推荐内容。
    核心需求：根据 candidate_apartments 回答用户原始问题。
    """
    print("\n---开始生成推荐小区的回答---")

    # 1. 从后往前找最新的用户消息 (保留，获取回答目标)
    user_original_question = ""
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, 'type') and msg.type == "human":
             user_original_question = msg.content
             break
        elif isinstance(msg, dict) and msg.get('type') == 'human':
             user_original_question = msg.get('content', '')
             break

    if not user_original_question:
        print("错误：在消息历史中未能找到用户的原始问题。")
        # 假设 AIMessage 已经被正确导入或模拟
        ai_response = AIMessage(content="抱歉，我似乎没有捕捉到您的问题，能麻烦您再说一遍吗？")
        return {"messages": [ai_response]}
    # 2. 从state中获取检索到的相关context
    candidate_apartments_list = state.get("candidate_apartments", []) # 直接从 state 中获取列表
    if not isinstance(candidate_apartments_list, list):
        print(f"警告: state['candidate_apartments']['candidate_apartments'] 不是列表类型，而是 {type(candidate_apartments_list)}")
        candidate_apartments_list = [] # 如果不是列表，重置为空

    print(f"从 state 获取到候选小区 ({len(candidate_apartments_list)}个 文档).")


    # 3. 检查是否有候选小区可以推荐 (保留，无资料则无法回答)
    if not candidate_apartments_list: # 使用获取到的列表进行检查
        print("警告：没有找到推荐的小区列表，无法生成具体推荐。")
        # 假设 AIMessage 已经被正确导入或模拟
        ai_response = AIMessage(content=f"抱歉，根据您的问题“{user_original_question}”，我暂时没有找到符合条件的小区房源信息。您可以尝试换个方式描述您的需求。")
        return {"messages": [ai_response]}

    # 4. 格式化检索到的文档内容，用于构建提示 (恢复这一步，对LLM很重要)
    try:
        # 使用获取到的文档列表调用格式化函数
        # 假设 format_documents_for_prompt 函数能够正确处理 Document 对象列表
        formatted_docs = rec_format_documents_for_prompt(candidate_apartments_list) # 将列表传递给格式化函数
        if not formatted_docs:
             # 即使有候选小区，如果格式化后为空，也视为无法提供有效推荐
             print("警告：format_documents_for_prompt 生成了空的背景资料字符串。")
             # 假设 AIMessage 已经被正确导入或模拟
             ai_response = AIMessage(content=f"抱歉，根据您的问题“{user_original_question}”，我在组织推荐信息时遇到了问题。请您稍后再试。")
             return {"messages": [ai_response]}

    except Exception as e:
         print(f"错误：格式化背景资料时发生异常: {e}")
         # 假设 AIMessage 已经被正确导入或模拟
         ai_response = AIMessage(content=f"抱歉，根据您的问题“{user_original_question}”，我在处理房源信息时遇到了问题。请您稍后再试。")
         return {"messages": [ai_response]}

    # 5. 构建LLM的Human提示 (精简，只保留根据背景资料回答的部分)
    # Prompt 直接引导 LLM 根据用户问题和背景资料进行推荐
    human_prompt_for_llm = (
        f"用户原始问题是：“{user_original_question}”\n\n"
        f"请仔细阅读以下背景资料，根据用户原始问题，从背景资料中挑选出最符合用户需求的小区，并向用户进行清晰、准确的推荐。\n\n"
        f"背景资料：\n{formatted_docs}\n\n" # 使用格式化后的背景资料
        f"请用友好的语气，以条理清晰的方式，结合背景资料生成推荐内容。不要提及你是AI或语言模型，直接像房产顾问一样给出推荐。"
    )

    # 6. 构建并调用 LLM Chain 生成回答 (保留)
    # 假设 ChatPromptTemplate, llm, StrOutputParser 已经被正确导入或模拟
    answer_generation_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", generate_answer_system_prompt_text), # 你的系统提示文本
            ("human", human_prompt_for_llm),
        ]
    )

    # 假设 llm 和 StrOutputParser 已经像您之前那样定义好了
    answer_chain = answer_generation_prompt | llm | StrOutputParser()

    # 打印部分提示，方便调试
    print(f"提交给LLM生成答案的Human Prompt (部分内容): {human_prompt_for_llm[:500]}...") # 打印前500字符

    try:
        # invoke({}) 是正确的，因为所有变量都已嵌入到 prompt 消息中
        final_answer_content = answer_chain.invoke({})
        # 确保内容是字符串，处理模拟类返回对象的情况
        if hasattr(final_answer_content, 'content'):
             final_answer_content = final_answer_content.content
        final_answer_content = str(final_answer_content).strip() # 确保是字符串再strip

        # 检查 LLM 是否返回了空字符串 (保留)
        if not final_answer_content:
             print("警告：LLM返回了空的回答内容。")
             # 假设 AIMessage 已经被正确导入或模拟
             final_answer_content = f"抱歉，根据您的问题“{user_original_question}”和找到的房源信息，我在生成推荐时遇到了问题。请您稍后再试或换个方式提问。"

    except Exception as e:
        print(f"调用LLM生成答案时发生严重错误: {e}")
        # 打印更详细的错误信息
        import traceback
        traceback.print_exc()
        # 假设 AIMessage 已经被正确导入或模拟
        final_answer_content = "抱歉，我在为您生成回答时遇到了一些技术问题，请您稍后再试或换个方式提问。"

    # 7. 返回生成的回答 (保留)
    # 假设 AIMessage 已经被正确导入或模拟
    ai_response = AIMessage(content=final_answer_content)

    # Langgraph 的 add_messages 会自动将这个新消息追加到 state["messages"]
    return {"messages": [ai_response]}

def extract_preferences(state: RecommendationState):
    """
    Langgraph 节点：从最新的用户消息中提取用户偏好，并更新到 state["user_preferences"]。
    使用真实的 LLM 和 JsonOutputParser。
    """
    print("\n---开始提取用户偏好---")
    messages = state.get("messages", [])
    latest_human_message_content = None # 使用更明确的变量名
    latest_human_message_obj = None # 可能需要原始消息对象用于 chain 输入
    # 从后往前找最新的用户消息
    for msg in reversed(messages):
        # 兼容 LangChain 消息对象和字典格式
        if hasattr(msg, 'type') and msg.type == "human":
             latest_human_message_content = msg.content
             latest_human_message_obj = msg # 如果 msg 是对象，保存它
             break
        elif isinstance(msg, dict) and msg.get('type') == 'human':
             latest_human_message_content = msg.get('content', '')
             # 如果是字典，为了构建 Chain 输入，可能需要转换为 HumanMessage 对象
             # 确保这里使用的 HumanMessage 是导入的或模拟的 LangChain 兼容类
             try:
                 latest_human_message_obj = HumanMessage(content=latest_human_message_content)
             except NameError:
                 print("警告：无法创建 HumanMessage 对象，Chain 输入可能受限。确保 HumanMessage 类可用。")
                 latest_human_message_obj = {"type": "human", "content": latest_human_message_content} # 回退到字典格式


             if latest_human_message_content: # 确保内容不为空
                break
    if not latest_human_message_content:
        print("未找到最新用户消息内容，跳过偏好提取。")
        # 如果没有用户消息可供提取，返回原始 state
        return state

    # 确保有消息对象用于构建 Chain 输入
    if latest_human_message_obj is None:
         print("警告：找到了用户消息内容但无法构建消息对象，跳过偏好提取。")
         return state


    print(f"正在从用户消息中提取偏好: {latest_human_message_content}")
    # 构建并调用 LLM Chain 提取用户偏好
    extraction_prompt_template = """作为一个房产信息提取助手，请从以下用户消息中识别和提取用户的房产偏好。
    提取的偏好包括但不限于：
    区域 (例如：黄浦, 徐汇, 浦东)
    预算 (例如：总价500万, 单价8万/平, 租金1万/月)
    户型 (例如：三居室, 2室1厅, 大开间)
    特点 (例如：近地铁, 有学校, 精装修, 环境好, 新小区)，如果用户提到了多个特点，请将它们作为字符串列表。
    面积 (例如：90平米以上, 120平米左右)
    购房/租房 (例如：购房, 租房)

    请将提取到的信息组织成一个 JSON 对象。如果用户未明确提及某个偏好项，请不要在 JSON 中包含该项。
    如果用户消息中没有包含任何房产偏好信息，返回一个空的 JSON 对象 {{}}.
    请严格按照 JSON 格式输出，不要包含任何额外的文字或说明。

    示例输出 JSON:
    {{
      "区域": "黄浦",
      "预算": "总价500万",
      "户型": "三居室",
      "特点": ["近地铁", "有学校"],
      "购房/租房": "购房"
    }}

    请开始提取：
    {user_message}
    """

    # 假设 ChatPromptTemplate 可以从模板创建
    extraction_prompt = ChatPromptTemplate.from_template(extraction_prompt_template)

    # 假设 llm 和 JsonOutputParser 已经像您之前那样定义好了
    try:
        # 构建提取 Chain
        extraction_chain = extraction_prompt | llm | JsonOutputParser()

        # 调用 Chain 执行提取
        # Prompt Template 接收输入变量，这里将用户消息作为 user_message 变量传入
        extracted_preferences = extraction_chain.invoke({"user_message": latest_human_message_content})

        print(f"LLM 提取到的偏好: {extracted_preferences}")

        # LLM 可能返回非字典类型（如 None 或字符串），或者 JsonOutputParser 解析失败（尽管它本身会抛异常）
        # 添加一个检查确保结果是字典
        if not isinstance(extracted_preferences, dict):
             print(f"警告: LLM 或解析器返回的结果不是字典，而是 {type(extracted_preferences)}. 尝试转换为字典。")
             # 尝试从字符串解析，以防 JsonOutputParser 没用或 LLM 直接返回 JSON 字符串
             if isinstance(extracted_preferences, str):
                 try:
                      extracted_preferences = json.loads(extracted_preferences)
                      if not isinstance(extracted_preferences, dict): raise ValueError("Parsed JSON is not a dictionary")
                 except (json.JSONDecodeError, ValueError) as e:
                      print(f"警告: 尝试手动解析 LLM 输出字符串失败: {e}. 将使用空偏好。")
                      extracted_preferences = {} # 解析失败则使用空字典
             else:
                  extracted_preferences = {} # 如果不是字符串也不是字典，使用空字典


        # --- 合并新提取的偏好与 state 中已有的偏好 ---
        current_preferences = state.get("user_preferences", {})
        updated_preferences = {**current_preferences, **extracted_preferences}


        print(f"合并后的用户偏好: {updated_preferences}")

        # --- 更新 state ---
        state["user_preferences"] = updated_preferences
        return state # 返回更新后的 state

    except Exception as e:
        print(f"提取用户偏好时发生错误: {e}")
        traceback.print_exc() # 打印完整的错误堆栈
        # 提取偏好失败通常不应该中断整个流程，记录错误并返回原始 state
        # 这样后续节点（如推荐）可以使用 state 中已有的偏好信息继续
        return state

# Langgraph 节点：生成简单回复（用于欢迎、感谢或无法识别的意图）
simple_response_system_prompt_text_v2 = """你是一位耐心、专业且乐于助人的房产顾问。
你正在处理用户的简短输入或与房产查询/推荐功能不直接相关的输入。
请保持你的房产顾问身份，并根据用户的输入给出简短、友好的回复。
如果用户的输入是一个问候或感谢，请自然地回应。
如果用户的输入与房产无关，请礼貌地将对话引导回房产话题，例如询问他们是否需要查询或推荐小区。
"""

simple_response_prompt_v2 = ChatPromptTemplate.from_messages([
    ("system", simple_response_system_prompt_text_v2),
    ("human", "{user_input}"),
])


# 修改后的 generate_simple_response 节点
def generate_simple_response(state: RecommendationState):
    """
    为通用情况（非特定查询、非推荐，即意图为欢迎、感谢或其他时）生成简单回复。
    确保 LLM 知道自己的房产助手角色。
    """
    print("\n---生成通用简单回复---")
    # 虽然函数内部不直接区分意图了，但打印出来有助于调试
    intent = state.get("intent", "未知")
    print(f"路由到的意图类别: '{intent}'. 生成通用回复。")


    last_user_message_content = ""
    # 从后往前找到最新的用户消息
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, 'type') and msg.type == "human":
             last_user_message_content = msg.content
             break
        elif isinstance(msg, dict) and msg.get('type') == 'human':
             last_user_message_content = msg.get('content', '')
             if last_user_message_content:
                break

    if not last_user_message_content:
         print("警告: 未找到用户消息内容，使用通用默认回复。")
         # 如果找不到用户消息，提供一个默认的欢迎语
         ai_response = AIMessage(content="您好！我是您的房产助手，请问有什么可以帮您查询或推荐的房产小区吗？")
         return {"messages": [ai_response]}

    print(f"用户输入: '{last_user_message_content[:50]}...'")

    try:
        # 构建并调用使用简化版提示模板的 LLM Chain
        simple_response_chain = simple_response_prompt_v2 | llm | StrOutputParser() # 使用 StrOutputParser 获取纯文本回复

        # Invoke Chain，只传入 user_input 变量
        response_content = simple_response_chain.invoke({
            "user_input": last_user_message_content
        })

        # 确保 LLM 返回的内容是字符串并去除首尾空白
        if hasattr(response_content, 'content'):
             response_content = response_content.content
        response_content = str(response_content).strip()

        # 如果 LLM 返回空内容，提供一个默认回复
        if not response_content:
             print("警告: LLM 生成了空的简单回复。使用通用默认回复。")
             response_content = "您好！我是您的房产助手，请问有什么可以帮您查询或推荐的房产小区吗？"

    except Exception as e:
        print(f"调用LLM生成简单回复时发生错误: {e}")
        traceback.print_exc() # 打印完整的错误栈
        response_content = "抱歉，我在生成回复时遇到了一些问题，请您稍后再试。"


    # 将生成的回复添加到消息历史中
    ai_response = AIMessage(content=response_content)
    return {"messages": [ai_response]}


# --- Router Definition ---
def route_based_on_intent(state: RecommendationState) -> str:
    """
    路由器函数：根据识别出的用户意图决定下一个节点。
    """
    print(f"\n---根据意图路由，当前意图: {state.get('intent')}---")
    intent = state.get("intent")
    if intent == "查询特定小区信息":
        return "extract_apartment_name"
    elif intent == "请求小区推荐":
        return "extract_preferences"
    # 将其他意图（包括“欢迎”和“感谢”）路由到生成简单回复的节点
    else:
        print(f"意图 '{intent}' 未匹配特定流程，路由到 generate_simple_response 节点。")
        return "generate_simple_response"


# --- Graph Definition ---

# 定义 Graph 工作流，使用 RecommendationState 作为状态
workflow = StateGraph(RecommendationState)

# 添加各个函数作为 Graph 的节点
workflow.add_node("identify_user_intent", identify_user_intent)
workflow.add_node("extract_apartment_name", extract_apartment_name)
workflow.add_node("retriever_specific_apartment_info", retriever_specific_apartment_info)
workflow.add_node("generate_specific_apartment_answer", generate_specific_apartment_answer)
workflow.add_node("extract_preferences", extract_preferences)
workflow.add_node("recommend_apartment", recommend_apartment)
workflow.add_node("generate_recommendation_answer", generate_recommendation_answer)
workflow.add_node("generate_simple_response", generate_simple_response) # 添加处理简单回复的节点


# 设置 Graph 的入口点为意图识别节点
workflow.set_entry_point("identify_user_intent")

# 添加边来连接节点

# 从入口点 START 到意图识别节点
# workflow.add_edge(START, "identify_user_intent")

# 从意图识别节点到后续节点的条件路由
# 根据 route_based_on_intent 函数的返回值决定下一个节点
workflow.add_conditional_edges(
    "identify_user_intent", # 当前节点
    route_based_on_intent, # 路由器函数
    { # 路由器函数返回的值与下一个节点名称的映射
        "extract_apartment_name": "extract_apartment_name",
        "extract_preferences": "extract_preferences",
        "generate_simple_response": "generate_simple_response" # 添加路由到简单回复节点
    }
)

# 特定小区查询流程的边
workflow.add_edge("extract_apartment_name", "retriever_specific_apartment_info")
workflow.add_edge("retriever_specific_apartment_info", "generate_specific_apartment_answer")

# 小区推荐流程的边
workflow.add_edge("extract_preferences", "recommend_apartment")
workflow.add_edge("recommend_apartment", "generate_recommendation_answer")

# 从生成回答的节点到流程结束 END
workflow.add_edge("generate_specific_apartment_answer", END)
workflow.add_edge("generate_recommendation_answer", END)
workflow.add_edge("generate_simple_response", END) # 从简单回复节点到 END

# 编译 Graph
app = workflow.compile()

print("--- Running the graph with a specific query ---")
inputs_query = {"messages": [HumanMessage(content="瑞南新苑这个小区在哪里，是什么时候建造的，我想了解一下。")]}
try:
    # 使用 stream 可以看到每个步骤的状态变化
    for output in app.stream(inputs_query, {"recursion_limit": 10}): # recursion_limit 防止无限循环
        for key, value in output.items():
            print(f"Node '{key}': {value}")
except Exception as e:
    print(f"Error running graph: {e}")
    traceback.print_exc()
print("\n---\n")
