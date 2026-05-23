import argparse
from utils import load_config, get_logger
from preprocessing import clean_text
from feature_extraction import TextTokenizer
from cnn_model import CNNFeatureExtractor
from svm_model import SVMClassifier

logger = get_logger("predict")

def predict_single_text(text):
    config = load_config()
    
    # 1. Làm sạch văn bản đầu vào
    cleaned = clean_text(text)
    logger.info(f"Văn bản gốc: '{text}'")
    logger.info(f"Văn bản sau làm sạch: '{cleaned}'")
    
    if not cleaned:
        logger.warning("Văn bản trống rỗng sau khi làm sạch! Mặc định là HAM.")
        return "HAM", 0.0
        
    # 2. Tải Tokenizer và mã hóa chuỗi
    tokenizer_path = config["paths"]["tokenizer_path"]
    tokenizer = TextTokenizer()
    tokenizer.load(tokenizer_path)
    sequence = tokenizer.texts_to_sequences([cleaned])
    
    # 3. Trích xuất đặc trưng bằng CNN
    extractor = CNNFeatureExtractor()
    extractor.load_model()
    features = extractor.extract_features(sequence)
    
    # 4. Dự đoán bằng mô hình SVM
    svm_model_path = config["paths"]["svm_model_path"]
    svm_clf = SVMClassifier()
    svm_clf.load(svm_model_path)
    
    pred_class = svm_clf.predict(features)[0]
    pred_probs = svm_clf.predict_proba(features)[0] # Mảng dạng [prob_ham, prob_spam]
    
    label = "SPAM" if pred_class == 1 else "HAM"
    spam_prob = pred_probs[1]
    
    print("\n" + "="*50)
    print("KẾT QUẢ DỰ ĐOÁN TIN NHẮN SPAM")
    print("="*50)
    print(f"Nội dung: {text}")
    print(f"Phân loại: {label}")
    print(f"Xác suất Spam: {spam_prob:.4%}")
    print("="*50 + "\n")
    
    return label, spam_prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dự đoán tin nhắn SMS có phải spam hay không.")
    parser.add_argument("--text", type=str, required=True, help="Nội dung tin nhắn cần kiểm tra.")
    args = parser.parse_args()
    
    predict_single_text(args.text)
