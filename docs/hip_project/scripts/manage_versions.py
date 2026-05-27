import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.version_control import VersionControl

def main():
    # 初始化版本控制
    vc = VersionControl('configs/version_control.yaml')
    
    # 列出当前所有版本
    print("当前版本信息：")
    versions = vc.list_versions()
    
    print("\n模型版本：")
    for model in versions['models']:
        print(f"- {model['name']} ({model['date']})")
        
    print("\n数据版本：")
    for data in versions['data']:
        print(f"- {data['version']} ({data['date']})")
        
    print("\n训练配置版本：")
    for config in versions['training_config']:
        print(f"- {config['version']} ({config['date']})")

if __name__ == "__main__":
    main() 