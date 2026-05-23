import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils import load_config, get_logger
from feature_extraction import TextTokenizer
from cnn_model import CNNFeatureExtractor
from svm_model import SVMClassifier

logger = get_logger("evaluate")

def evaluate_models():
    config = load_config()
    
    # 1. Đọc dữ liệu test
    split_dir = config["paths"]["split_dir"]
    test_csv = os.path.join(split_dir, "test.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Không tìm thấy dữ liệu test tại {test_csv}. Vui lòng chạy train.py trước.")
        
    test_df = pd.read_csv(test_csv)
    test_df['clean_text'] = test_df['clean_text'].fillna('')
    
    X_test_text = test_df['clean_text'].tolist()
    y_test = test_df['label_code'].values
    
    # 2. Tải Tokenizer và mã hóa dữ liệu test
    tokenizer_path = config["paths"]["tokenizer_path"]
    tokenizer = TextTokenizer()
    tokenizer.load(tokenizer_path)
    X_test = tokenizer.texts_to_sequences(X_test_text)
    
    # 3. Dự đoán bằng mô hình CNN thuần
    logger.info("Đang tải mô hình CNN để dự đoán thử nghiệm...")
    extractor = CNNFeatureExtractor()
    extractor.load_model()
    
    logger.info("Dự đoán bằng mô hình CNN thuần...")
    y_pred_cnn_prob = extractor.model.predict(X_test)
    y_pred_cnn = (y_pred_cnn_prob > 0.5).astype(int).flatten()
    
    # 4. Trích xuất đặc trưng và dự đoán bằng mô hình lai CNN-SVM
    logger.info("Trích xuất đặc trưng tập Test bằng CNN...")
    X_test_features = extractor.extract_features(X_test)
    
    logger.info("Đang tải mô hình SVM để dự đoán thử nghiệm...")
    svm_model_path = config["paths"]["svm_model_path"]
    svm_clf = SVMClassifier()
    svm_clf.load(svm_model_path)
    
    logger.info("Dự đoán bằng mô hình lai CNN-SVM...")
    y_pred_hybrid = svm_clf.predict(X_test_features)
    
    # 5. Tính toán các chỉ số đánh giá chi tiết
    metrics = {
        "CNN thuần": {
            "Accuracy": accuracy_score(y_test, y_pred_cnn),
            "Precision": precision_score(y_test, y_pred_cnn),
            "Recall": recall_score(y_test, y_pred_cnn),
            "F1-Score": f1_score(y_test, y_pred_cnn)
        },
        "Lai CNN-SVM": {
            "Accuracy": accuracy_score(y_test, y_pred_hybrid),
            "Precision": precision_score(y_test, y_pred_hybrid),
            "Recall": recall_score(y_test, y_pred_hybrid),
            "F1-Score": f1_score(y_test, y_pred_hybrid)
        }
    }
    
    # In bảng so sánh đẹp mắt
    metrics_df = pd.DataFrame(metrics).T
    print("\n" + "="*55)
    print("BẢNG SO SÁNH HIỆU NĂNG MÔ HÌNH TRÊN TẬP TEST")
    print("="*55)
    print(metrics_df.to_string(formatters={
        "Accuracy": "{:.4f}".format,
        "Precision": "{:.4f}".format,
        "Recall": "{:.4f}".format,
        "F1-Score": "{:.4f}".format
    }))
    print("="*55 + "\n")
    
    # In báo cáo phân loại cho mô hình lai
    logger.info("Báo cáo phân loại chi tiết (Classification Report) của mô hình lai CNN-SVM:")
    print(classification_report(y_test, y_pred_hybrid, target_names=['Ham', 'Spam']))
    
    # 6. Vẽ và lưu Ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred_hybrid)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                annot_kws={"size": 14, "weight": "bold"})
    plt.title('Ma trận nhầm lẫn - Mô hình lai CNN-SVM', fontsize=14, pad=15, fontweight='bold')
    plt.ylabel('Thực tế (True Label)', fontsize=12)
    plt.xlabel('Dự đoán (Predicted Label)', fontsize=12)
    plt.tight_layout()
    
    models_dir = config["paths"]["models_dir"]
    cm_path = os.path.join(models_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info(f"Đã lưu biểu đồ ma trận nhầm lẫn tại: {cm_path}")

if __name__ == "__main__":
    evaluate_models()
