"""
RAG系统配置文件
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, Any
import os
from pathlib import Path
from dotenv import load_dotenv

# 固定读取项目根目录的模型配置，不依赖终端当前目录。
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

@dataclass
class RAGConfig:
    """RAG系统配置类"""

    # 路径配置
    data_path: str = "../../data/C8/cook"
    index_save_path: str = "./vector_index"

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = field(default_factory=lambda: os.environ["model_name"])

    # 检索配置
    top_k: int = 3

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

# 默认配置实例
DEFAULT_CONFIG = RAGConfig()
