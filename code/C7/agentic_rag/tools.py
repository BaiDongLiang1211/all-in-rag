"""
Agentic RAG 工具集

定义了Agent可调用的检索工具，包括：
- VectorSearchTool: 向量数据库语义检索
- WebSearchTool: 互联网实时搜索

每个工具的 description 对 Agent 的工具选择决策至关重要。
"""

from typing import List, Optional
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_chroma import Chroma
import os
from pathlib import Path
from dotenv import load_dotenv

# 固定读取项目根目录的模型配置，不依赖终端当前目录。
load_dotenv(Path(__file__).resolve().parents[3] / ".env")


# ============================================================
# 工具1：向量数据库检索
# ============================================================
class VectorSearchTool:
    """
    封装向量数据库检索为Agent可调用的工具。

    设计要点：
    1. tool description 必须清晰说明适用/不适用场景
    2. 逐篇返回文档，便于Agent累计证据和去重
    3. 支持自定义返回数量k
    """

    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        self.search = tool("vector_search")(self._search)

    def _search(self, query: str, k: int = 5) -> List[Document]:
        """
        在本地向量知识库中检索与查询语义相关的文档。

        适用场景：
        - 查询需要从知识库中获取特定事实或概念
        - 问题涉及已有文档中的内容
        - 需要精确的文本片段作为答案依据

        不适用场景：
        - 需要最新实时信息（请使用 web_search 工具）
        - 需要结构化关系查询（请使用 kg_query 工具）

        Args:
            query: 经过优化的查询文本（建议包含关键词）
            k: 返回文档数量，默认5

        Returns:
            文档列表，距离分数保存在metadata中；没有结果时返回空列表
        """
        docs = self.vectorstore.similarity_search_with_score(query, k=k)
        # 保留全文；不将查询相关的距离或排名拼入正文，以免影响去重。
        return [
            Document(page_content=doc.page_content,
                     metadata={**doc.metadata, "distance": float(score)})
            for doc, score in docs
        ]


# ============================================================
# 工具2：Web搜索
# ============================================================
@tool("web_search")
def _web_search(query: str) -> List[Document]:
    """在互联网上搜索最新新闻、实时数据和近期事件。输入简洁的搜索关键词。"""
    from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

    result = TavilySearchAPIWrapper().raw_results(
        query=query, max_results=3, search_depth="advanced", include_answer=False
    )
    # 只返回网页证据，不把搜索服务生成的答案或响应元数据作为证据。
    documents = []
    seen = set()
    for item in result.get("results", []):
        content = (item.get("content") or "").strip()
        source = item.get("url") or ""
        identity = source or content
        if not content or identity in seen:
            continue
        seen.add(identity)
        documents.append(Document(
            page_content=content[:2000],
            metadata={"source": source, "title": item.get("title", ""), "kind": "web"},
        ))
    return documents


web_search_tool = _web_search if os.getenv("TAVILY_API_KEY") else None


# ============================================================
# 知识库构建工具
# ============================================================
def build_knowledge_base(
    documents: List[str],
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5",
    collection_name: str = "agentic_rag_demo",
    persist_directory: Optional[str] = None,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> Chroma:
    """
    构建向量知识库。

    Args:
        documents: 文档文本列表
        embedding_model_name: 本地 HuggingFace 嵌入模型名称
        collection_name: Chroma 集合名称
        persist_directory: 持久化目录（None则使用内存模式）
        chunk_size: 分块大小
        chunk_overlap: 分块重叠大小

    Returns:
        Chroma 向量存储实例
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    docs = [Document(page_content=text, metadata={"source": f"{collection_name}/原文{i}", "kind": "local"})
            for i, text in enumerate(documents, 1)]
    chunks = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    print(f"✅ 知识库构建完成: {len(chunks)} 个文本块, collection={collection_name}")
    return vectorstore
