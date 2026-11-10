from typing import List
import os
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pathlib import Path
from dotenv import load_dotenv

# 固定读取项目根目录的模型配置，不依赖终端当前目录。
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# 初始化 LLM
llm = ChatOpenAI(
    base_url=os.environ["DASHSCOPE_BASE_URL"],
    extra_body={"enable_thinking": False},
    model=os.environ["model_name"],
    api_key=os.environ["DASHSCOPE_API_KEY"]
)

# 1. 定义数据结构
class PersonInfo(BaseModel):
    name: str = Field(description="人物姓名")
    age: int = Field(description="人物年龄")
    skills: List[str] = Field(description="技能列表")

# 2. 创建解析器
parser = PydanticOutputParser(pydantic_object=PersonInfo)

# 3. 创建提示模板
prompt = PromptTemplate(
    template="请根据以下文本提取信息。\n{format_instructions}\n{text}\n",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# # 打印格式指令
# print("\n--- Format Instructions ---")
# print(parser.get_format_instructions())
# print("--------------------------\n")

# 4. 创建处理链
chain = prompt | llm | parser

# 5. 定义输入文本并执行调用链
text = "张三今年30岁，他擅长Python和Go语言。"
result = chain.invoke({"text": text})

# 6. 打印结果
print("\n--- 解析结果 ---")
print(f"结果类型: {type(result)}")
print(result)
print("--------------------\n")

print(f"姓名: {result.name}")
print(f"年龄: {result.age}")
print(f"技能: {result.skills}")
