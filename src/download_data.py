import os
import zipfile
import urllib.request
from utils import load_config, get_logger

logger = get_logger("download_data")

def download_and_extract():
    config = load_config()
    raw_dir = config["paths"]["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = os.path.join(raw_dir, "smsspamcollection.zip")
    
    logger.info(f"Đang tải dữ liệu từ {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        logger.info(f"Đã tải xong tệp zip về: {zip_path}")
        
        logger.info("Đang giải nén tập tin...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        logger.info(f"Đã giải nén dữ liệu vào thư mục: {raw_dir}")
        
        # Xóa file zip sau khi giải nén để tiết kiệm dung lượng
        os.remove(zip_path)
        logger.info("Đã dọn dẹp tệp zip.")
        
    except Exception as e:
        logger.error(f"Lỗi khi tải hoặc giải nén dữ liệu: {e}")
        raise e

if __name__ == "__main__":
    download_and_extract()
