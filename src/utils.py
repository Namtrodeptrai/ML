import os
import yaml
import pickle
import logging

def get_logger(name="spam_detection"):
    """
    Tạo logger hiển thị thông tin ra console.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def load_config(config_path=None):
    """
    Đọc cấu hình từ tệp YAML và tự động chuyển các đường dẫn tương đối thành tuyệt đối.
    """
    # Xác định thư mục gốc của dự án (thư mục cha của src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if config_path is None:
        config_path = os.path.join(project_root, "config.yaml")
    elif not os.path.isabs(config_path):
        config_path = os.path.abspath(os.path.join(project_root, config_path))
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # Chuyển đổi các đường dẫn tương đối trong config["paths"] thành tuyệt đối
    for key, path in config["paths"].items():
        if not os.path.isabs(path):
            config["paths"][key] = os.path.abspath(os.path.join(project_root, path))
            
    return config

def save_pickle(obj, filepath):
    """
    Lưu đối tượng Python thành tệp pickle.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filepath):
    """
    Tải đối tượng Python từ tệp pickle.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)
