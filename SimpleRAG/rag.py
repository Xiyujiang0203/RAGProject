from operator import itemgetter
from pathlib import Path
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
# 1.设置模型
# LLM配置：使用自定义API端点和gemini-2.5-flash模型
api_key = os.getenv("OPENAI_API_KEY")
base_url = "https://c-z0-api-01.hash070.com/v1"
llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=api_key,
    base_url=base_url
)

# Embedding模型：使用本地下载的BAAI/bge-m3模型
# 获取脚本所在目录，然后查找项目根目录下的model文件夹
script_dir = Path(__file__).parent
project_root = script_dir.parent if script_dir.name == "SimpleRAG" else script_dir
base_model_path = project_root / "model" / "bge-m3"

# 查找实际模型目录（ModelScope 可能在子目录中创建模型）
def find_model_directory(base_path):
    """递归查找包含模型文件的目录"""
    if not base_path.exists():
        return None
    
    # 检查当前目录是否有模型文件
    has_safetensors = any(base_path.glob("*.safetensors"))
    has_bin = any(base_path.glob("*.bin")) or (base_path / "pytorch_model.bin").exists()
    has_config = (base_path / "config.json").exists()
    
    if (has_safetensors or has_bin) and has_config:
        return base_path
    
    # 递归查找子目录（ModelScope 可能创建 BAAI/bge-m3 子目录）
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            result = find_model_directory(subdir)
            if result:
                return result
    
    return None

# 检查本地模型文件（支持 safetensors 和 bin 格式）
use_local = False
model_path = find_model_directory(base_model_path)

if model_path:
    # 检查是否有模型文件（优先 safetensors，其次 bin）
    has_safetensors = any(model_path.glob("*.safetensors"))
    has_bin = any(model_path.glob("*.bin")) or (model_path / "pytorch_model.bin").exists()
    has_config = (model_path / "config.json").exists()
    
    if has_safetensors:
        use_local = True
        print(f"使用本地模型: {model_path} (safetensors 格式)")
    elif has_bin and has_config:
        use_local = True
        print(f"使用本地模型: {model_path} (.bin 格式)")
        print("警告: .bin 格式需要 torch >= 2.6，如果遇到错误请升级 torch 或使用 safetensors 格式")
    else:
        print(f"警告: 找到模型目录但文件不完整，将使用在线模型")
else:
    print(f"警告: 本地模型目录不存在或缺少模型文件，将使用在线模型")
    print(f"提示: 请运行 python SimpleRAG/download_model.py 下载模型")

if use_local:
    # 使用本地模型
    try:
        print(f"正在加载本地模型: {model_path}")
        embedding_model = HuggingFaceEmbeddings(
            model_name=str(model_path),
            model_kwargs={
                "device": "cpu",
                "trust_remote_code": True
            }
        )
        print("✅ 本地模型加载成功")
    except (ValueError, ImportError, RuntimeError) as e:
        error_msg = str(e).lower()
        if "torch" in error_msg and "2.6" in error_msg:
            print(f"⚠️  警告: 使用 .bin 格式需要 torch >= 2.6")
            print(f"正在回退到在线模型...")
        elif "sentence_transformers" in error_msg or "torchvision" in error_msg:
            print(f"⚠️  警告: 本地模型加载失败（可能是依赖问题）")
            print(f"正在回退到在线模型...")
        else:
            print(f"⚠️  警告: 本地模型加载失败: {e}")
            print(f"正在回退到在线模型...")
        use_local = False

if not use_local:
    # 使用在线模型（会自动使用 safetensors 格式）
    print("使用在线模型: BAAI/bge-m3 (safetensors 格式)")
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={
            "device": "cpu",
            "trust_remote_code": True
        }
    )

# 2.设置数据处理（加载、分块、存储、检索）
# 使用脚本所在目录下的 my_knowledge 目录
file_dir = script_dir / 'my_knowledge'
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)
vector_store = Chroma(embedding_function=embedding_model,persist_directory='./chroma_v3')
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""你是一个严谨的RAG助手。
    请根据以下提供的上下文信息来回答问题。
    如果上下文信息不足以回答问题，请直接说“根据提供的信息无法回答”。
    如果回答时间使用了上下文中的信息，在回答后输出使用了哪些上下文，
    上下文信息：
    {context}
    问题：{question}"""
)

# 3.设置Chain
chain = {"question":RunnablePassthrough()} | RunnablePassthrough.assign(context=itemgetter("question")|retriever)|  prompt_template | llm | StrOutputParser()

# 4.初始化知识库
# 指定编码为 UTF-8 以支持中文文件
docs = DirectoryLoader(
    str(file_dir),
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
).load()
docs = text_splitter.split_documents(docs) # 分块
vector_store.add_documents(docs) # 存储文档

print(chain.invoke("能飞多久"))