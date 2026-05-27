import os
import json
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def fix_json_file(file_path):
    # 检测文件编码
    encoding = detect_encoding(file_path)
    print(f"Processing {file_path} with encoding: {encoding}")
    
    try:
        # 读取文件
        with open(file_path, 'r', encoding=encoding) as f:
            data = json.load(f)
        
        # 重新写入为UTF-8
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Successfully fixed {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def main():
    raw_dir = "raw_images"
    for filename in os.listdir(raw_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(raw_dir, filename)
            fix_json_file(file_path)

if __name__ == "__main__":
    main()
    