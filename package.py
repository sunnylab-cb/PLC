#!/usr/bin/env python3
import os
import sys
import fnmatch

# --- 配置区 ---
SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")
OUTPUT_FILE = "codebase_prompt.md"

# 【白名单】：如果列表不为空，则只处理匹配这些模式的文件
# 支持写法：
# - "*.py"       (匹配所有目录下的 .py 文件)
# - "main.py"    (匹配名为 main.py 的文件)
# - "src/*.yaml" (匹配 src 目录下的 yaml 文件)
INCLUDE_ONLY_PATTERNS = [
    "*.py", 
    "*.yaml",
    # "Dockerfile", # 示例：添加具体文件
]

# 【黑名单】：排除的目录
EXCLUDE_DIRS = [
    "node_modules", ".git", "dist", "build", ".vscode", ".idea",
    "__pycache__", "venv", ".nuxt", "diaries", "runs", "output*", 
    "site-packages"
]

# 【黑名单】：排除的文件
EXCLUDE_FILES = [
    "*.log", "*.tmp", "*.lock", "*.map", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "*.DS_Store", "*.sqlite3", "*.db", "*.png", "*.ico", "*.jpg", "*.jpeg", "*.gif", "*.svg",
    "*.woff", "*.woff2", "*.ttf", "*.eot", "*.pth", "*.npy", "tokenizer.json"
]
# --- END 配置区 ---

def main():
    if not os.path.isdir(SOURCE_DIR):
        print(f"错误：源目录 '{SOURCE_DIR}' 不存在。")
        sys.exit(1)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # 1. 生成项目结构
        f.write("# 项目结构\n\n")
        f.write(f"项目 `{SOURCE_DIR}` 目录结构：\n```\n")
        f.write(f"{os.path.basename(SOURCE_DIR.rstrip(os.sep))}/\n")
        generate_tree(SOURCE_DIR, '', f)
        f.write("```\n\n---\n\n")

        # 2. 拼接代码内容
        f.write("# 代码内容\n\n")
        generate_code_content(f)

    print(f"✅ 完成！文件已生成至: {OUTPUT_FILE}")

def is_excluded(path, is_dir=False):
    """检查路径是否在黑名单中"""
    name = os.path.basename(path)
    
    # 检查目录黑名单
    if is_dir:
        for pattern in EXCLUDE_DIRS:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False
    
    # 检查文件黑名单
    for pattern in EXCLUDE_FILES:
        if fnmatch.fnmatch(name, pattern):
            return True
    return False

def is_included(filename, relative_path):
    """
    检查是否在白名单中
    策略：如果白名单为空，默认全部包含。
         如果白名单有值，需要匹配 文件名 OR 相对路径 任意一个。
    """
    if not INCLUDE_ONLY_PATTERNS:
        return True
    
    for pattern in INCLUDE_ONLY_PATTERNS:
        # 1. 匹配文件名 (例如: "*.py", "main.py")
        if fnmatch.fnmatch(filename, pattern):
            return True
        # 2. 匹配相对路径 (例如: "src/*.py", "tests/test_api.py")
        if fnmatch.fnmatch(relative_path, pattern):
            return True
            
    return False

def generate_tree(dir_path, prefix, file_obj):
    try:
        items = sorted(os.listdir(dir_path))
    except PermissionError:
        return

    # 过滤掉黑名单中的目录和文件
    items = [i for i in items if not is_excluded(os.path.join(dir_path, i), is_dir=os.path.isdir(os.path.join(dir_path, i)))]

    count = len(items)
    for i, item in enumerate(items):
        path = os.path.join(dir_path, item)
        is_last = (i == count - 1)
        connector = "└── " if is_last else "├── "
        
        file_obj.write(f"{prefix}{connector}{item}\n")
        
        if os.path.isdir(path):
            new_prefix = prefix + ("    " if is_last else "│   ")
            generate_tree(path, new_prefix, file_obj)

def generate_code_content(f):
    if INCLUDE_ONLY_PATTERNS:
        print(f"ℹ️  白名单模式已开启: {INCLUDE_ONLY_PATTERNS}")
    
    for root, dirs, files in os.walk(SOURCE_DIR):
        # 原地修改 dirs 列表以剪除黑名单目录（优化遍历速度）
        dirs[:] = [d for d in dirs if not is_excluded(os.path.join(root, d), is_dir=True)]
        
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, SOURCE_DIR)
            
            # 1. 检查黑名单
            if is_excluded(file_path, is_dir=False):
                continue
                
            # 2. 检查白名单 (核心修复部分)
            if not is_included(file, relative_path):
                continue
            
            # 写入内容
            f.write(f"## 文件: `{relative_path}`\n\n")
            ext = os.path.splitext(file)[1].lstrip('.')
            lang = 'python' if ext == 'py' else (ext or 'text')
            
            f.write(f"```{lang}\n")
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as code_file:
                    content = code_file.read()
                    # 防止空文件导致格式难看
                    if not content.strip():
                        f.write("# (Empty File)\n")
                    else:
                        f.write(content)
            except Exception as e:
                f.write(f"Error reading file: {e}\n")
            f.write("\n```\n\n")

if __name__ == "__main__":
    main()