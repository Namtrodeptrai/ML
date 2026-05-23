#!/bin/bash

# Thử tìm lệnh Python phù hợp
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[LỖI] Không tìm thấy Python cài đặt trên hệ thống."
    exit 1
fi

echo "=========================================================="
echo "   HỆ THỐNG PHÁT HIỆN SMS SPAM - MÔ HÌNH LAI CNN + SVM"
echo "=========================================================="
echo ""

# Kiểm tra sự tồn tại của thư mục .venv
if [ ! -d ".venv" ]; then
    echo "[HỆ THỐNG] Không tìm thấy môi trường ảo .venv. Tiến hành khởi tạo..."
    $PYTHON_CMD -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[LỖI] Tạo môi trường ảo thất bại."
        exit 1
    fi
    
    echo "[HỆ THỐNG] Khởi tạo môi trường ảo thành công."
    echo "[HỆ THỐNG] Đang kích hoạt môi trường ảo và cài đặt thư viện..."
    source .venv/bin/activate
    
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[LỖI] Cài đặt thư viện thất bại."
        exit 1
    fi
    echo "[HỆ THỐNG] Cài đặt thư viện hoàn tất."
else
    echo "[HỆ THỐNG] Đang kích hoạt môi trường ảo..."
    source .venv/bin/activate
fi

echo ""
echo "[HỆ THỐNG] Đang khởi chạy Máy chủ Web (Flask)..."
echo "[HỆ THỐNG] Trình duyệt sẽ tự động mở liên kết http://127.0.0.1:5000 sau vài giây..."
echo ""

# Tự động mở trình duyệt tùy hệ điều hành
if [ "$(uname)" == "Darwin" ]; then
    # macOS
    open "http://127.0.0.1:5000"
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Linux
    if command -v xdg-open &>/dev/null; then
        xdg-open "http://127.0.0.1:5000"
    fi
fi

# Chạy ứng dụng Flask
python src/app.py
