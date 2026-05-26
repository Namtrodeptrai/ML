import os
import sys
from flask import Flask, request, jsonify, render_template

# Đảm bảo import được các module từ src/
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils import load_config, get_logger
from preprocessing import clean_text
from feature_extraction import TextTokenizer
from cnn_model import CNNFeatureExtractor
from svm_model import SVMClassifier

logger = get_logger("app_server")

# Cấu hình thư mục templates tĩnh cho giao diện web
app = Flask(__name__, template_folder="../templates")
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Khai báo các mô hình học máy toàn cục (Singleton)
tokenizer = None
extractor = None
svm_clf = None
config = None

def init_models():
    """
    Tải sẵn các mô hình vào RAM một lần duy nhất khi khởi chạy máy chủ.
    """
    global tokenizer, extractor, svm_clf, config
    logger.info("Bắt đầu tải các mô hình học máy vào bộ nhớ...")
    try:
        config = load_config()
        
        # 1. Tải Tokenizer
        tokenizer_path = config["paths"]["tokenizer_path"]
        tokenizer = TextTokenizer()
        tokenizer.load(tokenizer_path)
        
        # 2. Tải CNN Feature Extractor
        extractor = CNNFeatureExtractor()
        extractor.load_model()
        
        # 3. Tải SVM Classifier
        svm_model_path = config["paths"]["svm_model_path"]
        svm_clf = SVMClassifier()
        svm_clf.load(svm_model_path)
        
        logger.info("Đã tải xong toàn bộ mô hình! Máy chủ sẵn sàng nhận yêu cầu.")
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng khi tải mô hình: {e}")
        raise e

@app.route("/")
def index():
    """
    Trả về trang giao diện Frontend.
    """
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    API tiếp nhận văn bản và trả về kết quả dự đoán cùng xác suất dạng JSON.
    """
    if tokenizer is None or extractor is None or svm_clf is None:
        return jsonify({"error": "Các mô hình học máy chưa được tải hoàn toàn!"}), 500
        
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Dữ liệu đầu vào không hợp lệ. Cần truyền trường 'text'!"}), 400
        
    text = data["text"]
    cleaned = clean_text(text)
    
    if not cleaned:
        return jsonify({
            "text": text,
            "cleaned_text": "",
            "label": "HAM",
            "spam_probability": 0.0,
            "message": "Tin nhắn trống sau khi làm sạch. Mặc định là HAM."
        })
        
    try:
        # 1. Chuyển thành chuỗi số nguyên
        sequence = tokenizer.texts_to_sequences([cleaned])
        # 2. Trích xuất đặc trưng bằng CNN
        features = extractor.extract_features(sequence)
        # 3. Phân loại bằng SVM
        pred_class = svm_clf.predict(features)[0]
        pred_probs = svm_clf.predict_proba(features)[0] # [prob_ham, prob_spam]
        
        label = "SPAM" if pred_class == 1 else "HAM"
        spam_prob = float(pred_probs[1])
        
        return jsonify({
            "text": text,
            "cleaned_text": cleaned,
            "label": label,
            "spam_probability": spam_prob,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Lỗi khi xử lý dự đoán: {e}")
        return jsonify({"error": f"Lỗi nội bộ máy chủ: {str(e)}"}), 500

if __name__ == "__main__":
    init_models()
    logger.info("Đang khởi động Flask server tại http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
