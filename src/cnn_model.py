import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from utils import load_config, get_logger

logger = get_logger("cnn_model")

def build_cnn_model(vocab_size, embedding_dim, max_len, num_filters, kernel_size, dense_units, dropout_rate, learning_rate):
    """
    Xây dựng mô hình CNN 1D phân loại văn bản.
    """
    logger.info("Đang xây dựng kiến trúc mô hình CNN...")
    inputs = Input(shape=(max_len,), name="input_layer")
    
    # Tầng nhúng từ (Embedding Layer)
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="embedding_layer")(inputs)
    
    # Tầng tích chập 1 chiều (Conv1D)
    x = Conv1D(filters=num_filters, kernel_size=kernel_size, activation="relu", name="conv1d_layer")(x)
    
    # Tầng lấy mẫu toàn cục lớn nhất (GlobalMaxPooling1D)
    x = GlobalMaxPooling1D(name="global_max_pooling")(x)
    
    # Tầng Dense để biểu diễn đặc trưng rút gọn (Được dùng để trích xuất vector đặc trưng)
    x = Dense(dense_units, activation="relu", name="dense_features")(x)
    
    # Tầng loại bỏ bớt kết nối ngẫu nhiên (Dropout) để tránh quá khớp
    x = Dropout(dropout_rate, name="dropout_layer")(x)
    
    # Tầng phân loại nhị phân đầu ra (Sigmoid)
    outputs = Dense(1, activation="sigmoid", name="output_layer")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="CNN_Spam_Classifier")
    
    # Biên dịch mô hình
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    
    model.summary()
    return model

class CNNFeatureExtractor:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.model = None
        self.feature_model = None

    def load_model(self):
        """
        Khởi tạo mô hình và tải trọng số đã lưu.
        """
        vocab_size = self.config["preprocessing"]["vocab_size"]
        max_len = self.config["preprocessing"]["max_len"]
        embedding_dim = self.config["cnn"]["embedding_dim"]
        num_filters = self.config["cnn"]["num_filters"]
        kernel_size = self.config["cnn"]["kernel_size"]
        dense_units = self.config["cnn"]["dense_units"]
        dropout_rate = self.config["cnn"]["dropout_rate"]
        learning_rate = self.config["cnn"]["learning_rate"]
        weights_path = self.config["paths"]["cnn_weights_path"]

        # Xây dựng mô hình CNN
        self.model = build_cnn_model(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_len=max_len,
            num_filters=num_filters,
            kernel_size=kernel_size,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )

        # Tải trọng số nếu có sẵn
        if os.path.exists(weights_path):
            logger.info(f"Đang tải trọng số CNN từ {weights_path}...")
            self.model.load_weights(weights_path)
            logger.info("Đã tải trọng số thành công.")
        else:
            logger.warning(f"Không tìm thấy trọng số CNN tại {weights_path}. Mô hình sẽ sử dụng trọng số khởi tạo ngẫu nhiên.")

        # Tạo mô hình trích xuất đặc trưng từ tầng dense_features
        self.feature_model = Model(inputs=self.model.input, outputs=self.model.get_layer("dense_features").output)

    def extract_features(self, sequences):
        """
        Trích xuất vector đặc trưng cho các chuỗi số nguyên đầu vào.
        """
        if self.feature_model is None:
            self.load_model()
        logger.info(f"Đang trích xuất vector đặc trưng cho {len(sequences)} mẫu chuỗi...")
        features = self.feature_model.predict(sequences)
        return features
