import os
from sklearn.svm import SVC
from utils import load_config, get_logger, save_pickle, load_pickle

logger = get_logger("svm_model")

class SVMClassifier:
    def __init__(self, C=1.0, kernel="rbf", gamma="scale"):
        """
        Khởi tạo bộ phân loại SVM. Thiết lập probability=True để lấy xác suất dự đoán.
        """
        self.model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42)

    def fit(self, X, y):
        """
        Huấn luyện bộ phân loại SVM trên các vector đặc trưng.
        """
        logger.info(f"Đang huấn luyện mô hình SVM (C={self.model.C}, kernel={self.model.kernel}, gamma={self.model.gamma})...")
        self.model.fit(X, y)
        logger.info("Đã huấn luyện SVM thành công.")

    def predict(self, X):
        """
        Dự đoán lớp (0 hoặc 1).
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Dự đoán xác suất xác định lớp.
        """
        return self.model.predict_proba(X)

    def save(self, filepath):
        """
        Lưu mô hình SVM dạng pickle.
        """
        logger.info(f"Đang lưu mô hình SVM tại {filepath}...")
        save_pickle(self.model, filepath)

    def load(self, filepath):
        """
        Tải mô hình SVM từ tệp pickle.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy tệp mô hình SVM tại {filepath}")
        logger.info(f"Đang tải mô hình SVM từ {filepath}...")
        self.model = load_pickle(filepath)
