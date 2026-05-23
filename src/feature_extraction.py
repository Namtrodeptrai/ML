import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import save_pickle, load_pickle, load_config, get_logger

logger = get_logger("feature_extraction")

class TextTokenizer:
    def __init__(self, vocab_size=5000, max_len=80):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tokenizer = None

    def fit(self, texts):
        """
        Huấn luyện bộ Tokenizer trên danh sách văn bản.
        """
        logger.info(f"Đang huấn luyện Tokenizer với kích thước từ vựng tối đa: {self.vocab_size}")
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(texts)
        logger.info("Đã huấn luyện Tokenizer thành công.")

    def texts_to_sequences(self, texts):
        """
        Chuyển văn bản thành chuỗi số và thực hiện padding để có độ dài cố định.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer chưa được huấn luyện hoặc tải lên. Vui lòng gọi fit() hoặc load() trước.")
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        return padded_sequences

    def save(self, filepath):
        """
        Lưu Tokenizer ra file pickle.
        """
        logger.info(f"Đang lưu Tokenizer vào {filepath}...")
        save_pickle(self.tokenizer, filepath)

    def load(self, filepath):
        """
        Tải Tokenizer từ file pickle.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy tệp Tokenizer tại {filepath}")
        logger.info(f"Đang tải Tokenizer từ {filepath}...")
        self.tokenizer = load_pickle(filepath)
