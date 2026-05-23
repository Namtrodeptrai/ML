import sys
import os

# Thêm thư mục src vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from cnn_model import build_cnn_model

def test_cnn_model_shape():
    """
    Kiểm tra hình học kiến trúc của mạng CNN để đảm bảo các tham số hoạt động đúng.
    """
    vocab_size = 1000
    max_len = 50
    embedding_dim = 32
    num_filters = 16
    kernel_size = 3
    dense_units = 8
    dropout_rate = 0.2
    learning_rate = 0.001
    
    model = build_cnn_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_len=max_len,
        num_filters=num_filters,
        kernel_size=kernel_size,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )
    
    # Kiểm tra kích thước đầu vào và đầu ra
    assert model.input_shape == (None, max_len)
    assert model.output_shape == (None, 1)
    
    # Đảm bảo tầng dense_features tồn tại và có kích thước đầu ra đúng
    dense_layer = model.get_layer("dense_features")
    assert dense_layer.output.shape == (None, dense_units)
