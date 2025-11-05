"""
从 ModelScope 下载 BAAI/bge-m3 模型到本地 ./model/ 目录
"""
import os
from pathlib import Path

def download_bge_m3_model():
    """从 ModelScope 下载 BAAI/bge-m3 模型到  ./model/bge-m3/ 目录"""
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("❌ 未安装 modelscope 库")
        print("请运行: pip install modelscope")
        raise
    
    # ModelScope 上的模型路径
    model_name = "BAAI/bge-m3"
    # 获取脚本所在目录的父目录（项目根目录）
    script_dir = Path(__file__).parent
    project_root = script_dir.parent if script_dir.name == "SimpleRAG" else script_dir
    model_dir = project_root / "model" / "bge-m3"
    
    # 创建模型目录
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"开始从 ModelScope 下载模型: {model_name}")
    print(f"目标目录: {model_dir.absolute()}")
    
    try:
        # 从 ModelScope 下载模型到指定目录
        # ModelScope 的 snapshot_download 使用 model_id 参数
        model_path = snapshot_download(
            model_id=model_name,
            cache_dir=str(model_dir)
        )
        
        # 确保使用正确的路径
        final_model_dir = Path(model_path) if model_path else model_dir
        
        print(f"✅ 模型下载完成！")
        print(f"模型保存在: {final_model_dir.absolute()}")
        
        # 检查下载的文件格式
        if any(final_model_dir.glob("**/*.safetensors")):
            print("注意: 已下载 safetensors 格式")
        elif any(final_model_dir.glob("**/*.bin")):
            print("注意: 已下载 .bin 格式（PyTorch 格式）")
        else:
            print("注意: 请检查模型文件是否完整")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("提示: 确保已安装 modelscope: pip install modelscope")
        raise

if __name__ == "__main__":
    download_bge_m3_model()

