import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock

# Thêm thư mục src vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from predict import predict_single_text

@patch('predict.CNNFeatureExtractor')
@patch('predict.SVMClassifier')
@patch('predict.TextTokenizer')
def test_predict_single_text(mock_tokenizer_cls, mock_svm_cls, mock_extractor_cls):
    """
    Kiểm thử luồng logic của hàm dự đoán predict_single_text.
    Sử dụng unit test mock để giả lập việc gọi mô hình và tokenizer.
    """
    # Cấu hình mock cho Tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.texts_to_sequences.return_value = np.array([[1, 2, 3]])
    mock_tokenizer_cls.return_value = mock_tokenizer
    
    # Cấu hình mock cho CNN Feature Extractor
    mock_extractor = MagicMock()
    mock_extractor.extract_features.return_value = np.random.rand(1, 64)
    mock_extractor_cls.return_value = mock_extractor
    
    # Cấu hình mock cho SVM Classifier
    mock_svm = MagicMock()
    mock_svm.predict.return_value = np.array([1])                 # Dự đoán là SPAM (1)
    mock_svm.predict_proba.return_value = np.array([[0.1, 0.9]])   # Xác suất: Ham 10%, Spam 90%
    mock_svm_cls.return_value = mock_svm
    
    # Chạy hàm dự đoán với một chuỗi ngẫu nhiên
    label, spam_prob = predict_single_text("Claim a free prize now!")
    
    # Kiểm tra kết quả trả về đúng như mock thiết lập
    assert label == "SPAM"
    assert np.isclose(spam_prob, 0.9)
    
    # Đảm bảo các hàm mock tương ứng đã được gọi
    mock_tokenizer.texts_to_sequences.assert_called_once()
    mock_extractor.extract_features.assert_called_once()
    mock_svm.predict.assert_called_once()
    mock_svm.predict_proba.assert_called_once()
