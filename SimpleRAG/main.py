from openai import OpenAI
import os
from pathlib import Path

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key,base_url="https://c-z0-api-01.hash070.com/v1")

# 步骤1：实现一个简单的“关键词匹配”检索器
def retrival(query):
    context = ""
    
    # 0.遍历所有文件
    path_list = Path("my_knowledge").glob("*.txt")

    # 1.找到和问题相关的文件
    for path in path_list:
        if path.stem in query:
            # 2.相关文件内容读取出来
            context += path.read_text(encoding="utf-8")
            context += "\n\n"

    return context

# 步骤二：增强Query
def augemented(query,context=""):
    if not context:
        return f"请简要回答下面问题:{query}"
    else:
        prompt = f"""
        你是一个知识库助手，用户的问题是:{query}
        知识库内容如下:{context}
        请根据知识库内容回答用户问题，回答要求：
        1.回答要简洁明了，不要超过100字
        2.回答要准确，不要出现错误
        3.回答要符合知识库内容，不要出现知识库中没有的内容
        4.回答要符合用户问题，不要出现用户问题中没有的内容
        5.如果知识库信息不足以回答，请直接说：“根据已知信息无法回答”
        """
    return prompt

# 步骤三：生成回答
def generate_answer(prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # 不使用RAG模型
    query = input("请输入问题:")
    answer = generate_answer(query)
    print(answer)
    print("-"*100)
    
    # 使用RAG模型
    context = retrival(query)
    prompt = augemented(query,context)
    answer = generate_answer(prompt)
    print(answer)