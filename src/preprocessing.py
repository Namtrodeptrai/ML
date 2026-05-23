import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from utils import load_config, get_logger

logger = get_logger("preprocessing")

# Tải bộ từ dừng (stopwords) từ nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Đang tải dữ liệu stopwords từ NLTK...")
    nltk.download('stopwords')

def clean_text(text):
    """
    Hàm làm sạch văn bản SMS:
    - Chuyển thành chữ thường.
    - Loại bỏ ký tự đặc biệt và chữ số.
    - Loại bỏ khoảng trắng thừa.
    - Loại bỏ stop words tiếng Anh.
    """
    if not isinstance(text, str):
        return ""
    # Chuyển chữ thường
    text = text.lower()
    # Loại bỏ ký tự đặc biệt và chữ số
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Loại bỏ stop words
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    
    return ' '.join(words)

def preprocess_and_split():
    config = load_config()
    raw_filepath = config["paths"]["raw_filepath"]
    
    if not os.path.exists(raw_filepath):
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu thô tại: {raw_filepath}. Vui lòng chạy download_data.py trước.")
        
    logger.info(f"Đang đọc dữ liệu từ {raw_filepath}...")
    # Tệp SMSSpamCollection có cấu trúc dạng: label \t text (không có tiêu đề)
    df = pd.read_csv(raw_filepath, sep='\t', names=['label', 'text'], header=None)
    logger.info(f"Tổng số mẫu đọc được: {len(df)}")
    
    logger.info("Đang tiền xử lý văn bản...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Mã hóa nhãn: ham -> 0, spam -> 1
    df['label_code'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Loại bỏ các hàng có văn bản trống sau khi làm sạch
    df = df[df['clean_text'] != ""].reset_index(drop=True)
    
    # Chia tập dữ liệu: Train / Val / Test
    test_size = config["preprocessing"]["test_size"]
    val_size = config["preprocessing"]["val_size"]
    random_state = config["preprocessing"]["random_state"]
    
    # Tách tập test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label_code']
    )
    
    # Tỷ lệ val trên tập train_val còn lại
    val_ratio_adjusted = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio_adjusted, random_state=random_state, stratify=train_val_df['label_code']
    )
    
    logger.info(f"Số lượng mẫu - Tập Train: {len(train_df)}")
    logger.info(f"Số lượng mẫu - Tập Val: {len(val_df)}")
    logger.info(f"Số lượng mẫu - Tập Test: {len(test_df)}")
    
    # Lưu các tập dữ liệu
    split_dir = config["paths"]["split_dir"]
    os.makedirs(split_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(split_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(split_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(split_dir, "test.csv"), index=False)
    
    # Lưu toàn bộ dữ liệu sạch
    processed_dir = config["paths"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    df.to_csv(os.path.join(processed_dir, "clean_spam.csv"), index=False)
    
    logger.info("Đã tiền xử lý dữ liệu và chia tách thành công.")

if __name__ == "__main__":
    preprocess_and_split()
