import os
import yaml
import shutil
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VersionControl:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self):
        """加载版本控制配置文件"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _save_config(self):
        """保存版本控制配置"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
    def add_model_version(self, model_path, metrics=None, description=""):
        """添加新的模型版本"""
        model_name = os.path.basename(model_path)
        backup_path = os.path.join('backups', 'models', model_name)
        
        # 复制模型文件到备份目录
        shutil.copy2(model_path, backup_path)
        
        # 更新配置文件
        model_info = {
            'name': model_name,
            'path': backup_path,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'metrics': metrics or {},
            'description': description
        }
        
        self.config['version_control']['models'].append(model_info)
        self._save_config()
        logger.info(f"Added new model version: {model_name}")
        
    def add_data_version(self, data_path, stats=None, description=""):
        """添加新的数据版本"""
        version = f"v{len(self.config['version_control']['data']['versions']) + 1}.0"
        backup_path = os.path.join('backups', 'data', f"data_{version}")
        
        # 复制数据到备份目录
        if os.path.isdir(data_path):
            shutil.copytree(data_path, backup_path)
        else:
            shutil.copy2(data_path, backup_path)
            
        # 更新配置文件
        data_info = {
            'version': version,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'description': description,
            'stats': stats or {}
        }
        
        self.config['version_control']['data']['versions'].append(data_info)
        self.config['version_control']['data']['current_version'] = version
        self._save_config()
        logger.info(f"Added new data version: {version}")
        
    def get_latest_model(self):
        """获取最新的模型版本"""
        models = self.config['version_control']['models']
        if not models:
            return None
        return models[-1]
        
    def get_latest_data(self):
        """获取最新的数据版本"""
        return self.config['version_control']['data']['current_version']
        
    def list_versions(self):
        """列出所有版本信息"""
        return {
            'models': self.config['version_control']['models'],
            'data': self.config['version_control']['data']['versions'],
            'training_config': self.config['version_control']['training_config']['versions']
        } 