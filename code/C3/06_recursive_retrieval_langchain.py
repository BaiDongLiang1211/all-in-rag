import ast
import os
import re
from functools import partial
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_experimental.tools import PythonAstREPLTool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")

# 配置模型，复用项目 .env 中的模型名、API Key 和接口地址。
required_env = ("model_name", "DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL")
missing_env = [name for name in required_env if not os.getenv(name)]
if missing_env:
    raise ValueError(f"请在项目 .env 或环境变量中配置: {', '.join(missing_env)}")

llm = ChatOpenAI(
    model=os.environ["model_name"],
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url=os.environ["DASHSCOPE_BASE_URL"],
    temperature=0.0,
    max_tokens=2048,
    timeout=60,
    # 避免输出额度耗在思考内容中，导致传给 Pandas 的查询代码为空。
    extra_body={"enable_thinking": False},
)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


def execute_pandas(code, *, python_tool):
    """直接执行查询链生成的 Pandas 代码，不使用 Agent 决策循环。"""
    # 在执行前要求一条非空表达式，避免空代码或 import 进入执行器。
    code = code.strip()
    ast.parse(code, mode="eval")
    print(f"Pandas 查询代码:\n{code}")
    result = python_tool.invoke(code)
    if isinstance(result, str) and re.match(r"^\w*(?:Error|Exception):", result):
        raise RuntimeError(f"Pandas 查询执行失败：{result}")
    print(f"Pandas 查询结果:\n{result}")
    return str(result)


# 1. 加载数据，为每个工作表创建查询链和摘要文档。
excel_file = project_root / "data/C3/excel/movie.xlsx"
df_query_engines = {}
all_nodes = []

pandas_prompt = ChatPromptTemplate.from_template(
    "你正在查询工作表 {sheet_name} 的完整 DataFrame，变量名为 df。\n"
    "该表只包含 {year} 年的电影，年份已在选择工作表时筛选完成。\n"
    "编号是记录编号，不是年份；不要用编号或编号前缀筛选年份。\n"
    "列名和类型:\n{df_types}\n前五行:\n{df_head}\n"
    "问题: {query}\n"
    "df 和 pd 已经提供。只输出一条可求值的 Pandas 表达式，"
    "禁止 import/from、赋值、print、Markdown 和解释，不要修改 df。"
    "查询时使用完整 df。查找最小值或最大值对应的电影时，可使用目标列的 "
    "idxmin/idxmax 配合 df.loc，返回电影名称及被比较的数值，便于核对。"
)

with pd.ExcelFile(excel_file) as xls:
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        year = sheet_name.replace("年份_", "")

        # 对应 PandasQueryEngine：模型生成代码，然后直接调用执行工具。
        # 与原示例一样会执行模型生成的 Python 代码，用于本地可信表格示例。
        python_tool = PythonAstREPLTool(locals={"df": df, "pd": pd})
        query_engine = (
            pandas_prompt.partial(
                sheet_name=sheet_name,
                year=year,
                df_types=df.dtypes.to_string(),
                df_head=df.head().to_string(index=False),
            )
            | llm
            | StrOutputParser()
            | RunnableLambda(partial(execute_pandas, python_tool=python_tool))
        )

        # 对应 IndexNode：摘要用于检索，metadata 保存目标工作表标识。
        summary = f"这个表格包含了年份为 {year} 的电影信息，可以用来回答关于这一年电影的具体问题。"
        node = Document(page_content=summary, metadata={"index_id": sheet_name})
        all_nodes.append(node)
        df_query_engines[sheet_name] = query_engine

# 2. 创建顶层索引，只对摘要文档建立向量索引。
vector_index = FAISS.from_documents(all_nodes, embeddings)

# 3. 先检索摘要，再根据 index_id 调用对应的表格查询链。
vector_retriever = vector_index.as_retriever(search_kwargs={"k": 1})


def retrieve_table(query):
    node = vector_retriever.invoke(query)[0]
    index_id = node.metadata["index_id"]
    print(f"命中的工作表: {index_id}")
    result = df_query_engines[index_id].invoke({"query": query})
    return {"query": query, "context": result}


recursive_retriever = RunnableLambda(retrieve_table)

# 4. 创建查询链，将表格查询结果汇总为自然语言回答。
response_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个专业的数据分析助手。\n"
        "前面的 Pandas 查询已经成功执行，"
        "你收到的是查询结果，而不是需要执行的代码。\n"
        "请严格根据查询结果回答用户的问题。\n"
        "注意：\n"
        "1. Pandas Series中的 Name、dtype、索引信息不是错误。\n"
        "2. 如果结果包含电影名称和评分人数，"
        "直接给出电影名称及对应人数。\n"
        "3. 不要重新计算，不要猜测，不要忽略已有结果。\n"
        "4. 只有查询结果明确包含错误信息时，"
        "才能回答查询失败。"
    ),
    (
        "human",
        "用户问题：{query}\n"
        "Pandas 查询结果：\n{context}\n"
        "请给出最终答案。"
    )
])
query_engine = recursive_retriever | response_prompt | llm | StrOutputParser()

# 5. 执行查询。
query = "1994年评分人数最少的电影是哪一部？"
print(f"查询: {query}")
response = query_engine.invoke(query)
print(f"回答: {response}")
