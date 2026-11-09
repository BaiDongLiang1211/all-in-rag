import os
from pathlib import Path

from dotenv import load_dotenv


from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.llms.openai_like import OpenAILike

from llama_index.core.node_parser import TokenTextSplitter


def main():

    project_root = Path(__file__).resolve().parents[2]

    load_dotenv(project_root / ".env")


    # 配置全局大语言模型，通过 OpenAILike 调用兼容 OpenAI 的接口
    Settings.llm = OpenAILike(
        model=os.environ["model_name"],
        api_key=os.environ["DASHSCOPE_API_KEY"],
        api_base=os.environ["DASHSCOPE_BASE_URL"],
        is_chat_model=True,
        temperature=0.7,
        max_tokens=2048,
        context_window=8192,
    )

    print("正在加载嵌入模型……", flush=True)


    # 使用本地运行的中文嵌入模型
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5", device="cpu"
    )
    markdown_path = project_root / "data/C1/markdown/easy-rl-chapter1.md"


    # 加载 Markdown 文件，转换为 Document 对象
    docs = SimpleDirectoryReader(input_files=[str(markdown_path)]).load_data()
    print("正在构建索引……", flush=True)


    # 构建向量索引：每块 1024 token，相邻块重叠 200 token
    index = VectorStoreIndex.from_documents(
        docs, transformations=[TokenTextSplitter(chunk_size=1024, chunk_overlap=200)]
    )


    # 创建查询引擎，检索最相关的 3 个文本块
    query_engine = index.as_query_engine(similarity_top_k=3)

    # 查看查询引擎的提示词模板
    print(query_engine.get_prompts(), flush=True)


    # 根据检索到的上下文，调用大语言模型回答问题
    print(query_engine.query("文中举了哪些例子？"), flush=True)


if __name__ == "__main__":

    main()
