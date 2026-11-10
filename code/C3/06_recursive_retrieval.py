import os
import sys
from importlib.util import find_spec, module_from_spec
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.schema import IndexNode

# 本例仅使用 Pandas 查询功能，不加载 experimental.__init__ 中无关的
# Nudge/微调模块，避免其旧 Mistral SDK 导入方式与当前 SDK 3.x 冲突。
# 保留真实包的搜索路径，下面仍导入已安装的原版 PandasQueryEngine。
if "llama_index.experimental" not in sys.modules:
    experimental_spec = find_spec("llama_index.experimental")
    if experimental_spec is None:
        raise ModuleNotFoundError("请安装项目依赖 llama-index-experimental")
    sys.modules[experimental_spec.name] = module_from_spec(experimental_spec)

from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")

# 使用项目 .env 中的模型名、DashScope API Key 和兼容接口地址。
required_env = ("model_name", "DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL")
missing_env = [name for name in required_env if not os.getenv(name)]
if missing_env:
    raise ValueError(f"请在项目 .env 或环境变量中配置: {', '.join(missing_env)}")

llm = OpenAILike(
    model=os.environ["model_name"],
    api_key=os.environ["DASHSCOPE_API_KEY"],
    api_base=os.environ["DASHSCOPE_BASE_URL"],
    is_chat_model=True,
    temperature=0.0,
    max_tokens=2048,
    context_window=8192,
    # 当前 DeepSeek 模型的思考与回答共用输出额度；表格查询直接生成代码。
    additional_kwargs={"extra_body": {"enable_thinking": False}},
)
# 将上面创建的模型对象注册为当前进程中 LlamaIndex 的默认 LLM。
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# 1.加载数据并为每个工作表创建查询引擎和摘要节点
excel_file = project_root / "data/C3/excel/movie.xlsx"
xls = pd.ExcelFile(excel_file)

df_query_engines = {}
all_nodes = []

for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)
    year = sheet_name.replace('年份_', '')

    # 默认 df.head() 的字符串可能省略中间列，显式提供完整字段和样例。
    pandas_prompt = PromptTemplate(
        "你正在查询工作表 {sheet_name} 的完整 DataFrame，变量名为 df。\n"
        "该表只包含 {year} 年的电影，年份已在选择工作表时筛选完成。\n"
        "编号是记录编号，不是年份；不要用编号或编号前缀筛选年份。\n"
        "完整列名及类型：\n{df_schema}\n"
        "前五行样例（查询时应使用完整 df）：\n{table_preview}\n"
        "{instruction_str}\n"
        "df 和 pd 已经提供。只输出一条可求值的 Pandas 表达式，"
        "禁止 import/from、赋值、print、Markdown 和解释，不要修改 df。\n"
        "查找最小值或最大值对应的电影时，可使用目标列的 idxmin/idxmax "
        "配合 df.loc，返回电影名称及被比较的数值，便于核对。\n"
        "问题：{query_str}\n表达式："
    ).partial_format(
        sheet_name=sheet_name,
        year=year,
        df_schema=df.dtypes.to_string(),
        table_preview=df.head().to_string(index=False),
    )
    
    # 为当前工作表（DataFrame）创建一个 PandasQueryEngine
    query_engine = PandasQueryEngine(
        df=df, llm=Settings.llm, pandas_prompt=pandas_prompt, verbose=True
    )
    
    # 为当前工作表创建一个摘要节点（IndexNode）
    summary = f"这个表格包含了年份为 {year} 的电影信息，可以用来回答关于这一年电影的具体问题。"
    node = IndexNode(text=summary, index_id=sheet_name)
    all_nodes.append(node)
    
    # 存储工作表名称到其查询引擎的映射
    df_query_engines[sheet_name] = query_engine

# 2. 创建顶层索引（只包含摘要节点）
vector_index = VectorStoreIndex(all_nodes)

# 3. 创建递归检索器
# 3.1 创建顶层检索器，用于在摘要节点中检索
vector_retriever = vector_index.as_retriever(similarity_top_k=1)

# 3.2 创建递归检索器
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=df_query_engines,
    verbose=True,
)

# 4. 创建查询引擎
query_engine = RetrieverQueryEngine.from_args(recursive_retriever)

# 5. 执行查询
query = "1994年评分人数最少的电影是哪一部？"
print(f"查询: {query}")
# 显式分开检索与回答：表格执行错误应直接报告，不交给 LLM 当成查询结果。
retrieved_nodes = recursive_retriever.retrieve(query)
for retrieved_node in retrieved_nodes:
    if "There was an error running the output as Python code." in retrieved_node.node.get_content():
        raise RuntimeError("Pandas 查询执行失败，请检查上面打印的模型生成代码。")
response = query_engine.synthesize(query, nodes=retrieved_nodes)
print(f"回答: {response}")
