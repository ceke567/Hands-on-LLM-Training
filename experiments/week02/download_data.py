import os
import urllib.request
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
data_dir = repo_root / "data" / "raw"
data_dir.mkdir(parents=True, exist_ok=True)
dest_path = data_dir / "tinystories.txt"

url = "https://modelscope.cn/api/v1/datasets/AI-ModelScope/TinyStories/repo?Revision=master&FilePath=TinyStories-valid.txt"

print(f"=== 开始下载 TinyStories 数据集 (国内 ModelScope 高速源) ===")
print(f"目标保存路径: {dest_path}")

req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
    total_size = int(response.headers.get("Content-Length", 0))
    print(f"文件总大小: {total_size / (1024 * 1024):.2f} MB")
    
    downloaded = 0
    chunk_size = 1024 * 1024 # 1MB
    while True:
        chunk = response.read(chunk_size)
        if not chunk:
            break
        out_file.write(chunk)
        downloaded += len(chunk)
        percent = (downloaded / total_size) * 100 if total_size > 0 else 0
        print(f"\r下载进度: {downloaded / (1024 * 1024):.2f} MB / {total_size / (1024 * 1024):.2f} MB ({percent:.1f}%)", end="")

print("\n\n🎉 下载完成！正在校验数据...")

with open(dest_path, "r", encoding="utf-8", errors="ignore") as f:
    sample_text = f.read(500)
    f.seek(0)
    total_chars = sum(len(line) for line in f)

print(f"数据总字符数: {total_chars:,} 字符 (约 {total_chars / 1e6:.1f}M)")
print("\n--- [前 300 字符故事预览] ---")
print(sample_text[:300].strip())
print("----------------------------")
