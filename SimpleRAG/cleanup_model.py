"""
清理已下载模型中的 .bin, .pt, .pth 文件，只保留 safetensors 格式
解决 torch 版本兼容性问题
"""
from pathlib import Path

def cleanup_model_files():
    """清理模型目录中的 PyTorch 格式文件，只保留 safetensors"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent if script_dir.name == "SimpleRAG" else script_dir
    model_path = project_root / "model" / "bge-m3"
    
    if not model_path.exists():
        print(f"模型目录不存在: {model_path}")
        return
    
    # 查找并删除 .bin, .pt, .pth 文件
    patterns = ["*.bin", "*.pt", "*.pth"]
    deleted_count = 0
    
    for pattern in patterns:
        for file in model_path.glob(pattern):
            print(f"删除: {file.name}")
            file.unlink()
            deleted_count += 1
    
    if deleted_count == 0:
        print("没有找到需要清理的文件")
    else:
        print(f"✅ 已清理 {deleted_count} 个不兼容的文件")
        print("现在模型目录只包含 safetensors 格式文件")

if __name__ == "__main__":
    cleanup_model_files()

