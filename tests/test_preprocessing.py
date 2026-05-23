import sys
import os

# Thêm thư mục src vào sys.path để import các module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import clean_text

def test_clean_text():
    """
    Kiểm thử hàm làm sạch văn bản.
    """
    raw_text = "URGENT! Click here to win a FREE camera!!! No, really, it's true."
    cleaned = clean_text(raw_text)
    
    # Chữ cái hoa phải được chuyển thành thường
    assert "urgent" in cleaned
    assert "free" in cleaned
    
    # Ký tự đặc biệt và chữ số phải bị loại bỏ
    assert "!!!" not in cleaned
    assert "!" not in cleaned
    assert "," not in cleaned
    assert "." not in cleaned
    
    # Stop words tiếng Anh (như 'to', 'a', 'here') phải được lọc bỏ
    cleaned_words = cleaned.split()
    assert "to" not in cleaned_words
    assert "a" not in cleaned_words
    assert "here" not in cleaned_words
