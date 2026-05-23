import os
import pandas as pd
import numpy as np
from utils import load_config, get_logger
from download_data import download_and_extract
from preprocessing import preprocess_and_split
from feature_extraction import TextTokenizer
from cnn_model import build_cnn_model, CNNFeatureExtractor
from svm_model import SVMClassifier
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

logger = get_logger("train")

def run_pipeline():
    config = load_config()
    
    # 1. Kiểm tra và tải dữ liệu thô
    raw_filepath = config["paths"]["raw_filepath"]
    if not os.path.exists(raw_filepath):
        logger.info("Không tìm thấy dữ liệu thô tại đích. Bắt đầu tải dữ liệu...")
        download_and_extract()
        
    # 2. Tiền xử lý dữ liệu và chia tách
    split_dir = config["paths"]["split_dir"]
    train_csv = os.path.join(split_dir, "train.csv")
    val_csv = os.path.join(split_dir, "val.csv")
    test_csv = os.path.join(split_dir, "test.csv")
    
    if not (os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv)):
        logger.info("Không tìm thấy các tệp phân tách dữ liệu. Tiến hành tiền xử lý...")
        preprocess_and_split()
        
    # Đọc dữ liệu sạch
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    # Đảm bảo không có giá trị trống (NaN) do quá trình đọc file
    train_df['clean_text'] = train_df['clean_text'].fillna('')
    val_df['clean_text'] = val_df['clean_text'].fillna('')
    test_df['clean_text'] = test_df['clean_text'].fillna('')
    
    # 3. Khởi tạo và huấn luyện Tokenizer
    vocab_size = config["preprocessing"]["vocab_size"]
    max_len = config["preprocessing"]["max_len"]
    tokenizer_path = config["paths"]["tokenizer_path"]
    
    os.makedirs(config["paths"]["models_dir"], exist_ok=True)
    
    tokenizer = TextTokenizer(vocab_size=vocab_size, max_len=max_len)
    tokenizer.fit(train_df['clean_text'].tolist())
    tokenizer.save(tokenizer_path)
    
    # Chuyển đổi các tập dữ liệu thành dạng chuỗi số nguyên
    X_train = tokenizer.texts_to_sequences(train_df['clean_text'].tolist())
    X_val = tokenizer.texts_to_sequences(val_df['clean_text'].tolist())
    X_test = tokenizer.texts_to_sequences(test_df['clean_text'].tolist())
    
    y_train = train_df['label_code'].values
    y_val = val_df['label_code'].values
    y_test = test_df['label_code'].values
    
    # 4. Huấn luyện mô hình CNN làm nền tảng
    embedding_dim = config["cnn"]["embedding_dim"]
    num_filters = config["cnn"]["num_filters"]
    kernel_size = config["cnn"]["kernel_size"]
    dense_units = config["cnn"]["dense_units"]
    dropout_rate = config["cnn"]["dropout_rate"]
    learning_rate = config["cnn"]["learning_rate"]
    epochs = config["cnn"]["epochs"]
    batch_size = config["cnn"]["batch_size"]
    cnn_weights_path = config["paths"]["cnn_weights_path"]
    
    cnn_model = build_cnn_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_len=max_len,
        num_filters=num_filters,
        kernel_size=kernel_size,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )
    
    # Callbacks để tự động lưu trọng số tốt nhất và dừng sớm khi mô hình bão hòa
    checkpoint = ModelCheckpoint(
        filepath=cnn_weights_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    logger.info("Bắt đầu quá trình huấn luyện CNN...")
    cnn_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    logger.info("Huấn luyện CNN hoàn thành.")
    
    # 5. Sử dụng CNN đã huấn luyện để trích xuất đặc trưng cho SVM
    extractor = CNNFeatureExtractor()
    extractor.load_model()
    
    logger.info("Bắt đầu trích xuất đặc trưng từ mô hình CNN cho tập huấn luyện...")
    X_train_features = extractor.extract_features(X_train)
    logger.info(f"Đặc trưng trích xuất thành công. Kích thước tập Train mới: {X_train_features.shape}")
    
    # 6. Huấn luyện SVM trên đặc trưng của CNN
    svm_c = config["svm"]["C"]
    svm_kernel = config["svm"]["kernel"]
    svm_gamma = config["svm"]["gamma"]
    svm_model_path = config["paths"]["svm_model_path"]
    
    svm_clf = SVMClassifier(C=svm_c, kernel=svm_kernel, gamma=svm_gamma)
    svm_clf.fit(X_train_features, y_train)
    svm_clf.save(svm_model_path)
    
    logger.info("Toàn bộ quy trình huấn luyện tích hợp CNN-SVM đã hoàn thành thành công!")

if __name__ == "__main__":
    run_pipeline()
