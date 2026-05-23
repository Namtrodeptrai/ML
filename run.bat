@echo off
chcp 65001 > nul
title SMS Spam Detection System - CNN + SVM

echo ==========================================================
echo    HỆ THỐNG PHÁT HIỆN SMS SPAM - MÔ HÌNH LAI CNN + SVM
echo ==========================================================
echo.

:: Kiểm tra sự tồn tại của thư mục .venv
if not exist .venv (
    echo [HỆ THỐNG] Không tìm thấy môi trường ảo .venv. Tiến hành khởi tạo...
    python -m venv .venv
    if errorlevel 1 (
        echo [LỖI] Tạo môi trường ảo thất bại. Hãy chắc chắn rằng bạn đã cài đặt Python 3.8+.
        pause
        exit /b 1
    )
    
    echo [HỆ THỐNG] Khởi tạo môi trường ảo thành công.
    echo [HỆ THỐNG] Đang kích hoạt môi trường ảo và cài đặt các thư viện cần thiết...
    call .venv\Scripts\activate
    
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [LỖI] Cài đặt các thư viện cần thiết thất bại.
        pause
        exit /b 1
    )
    echo [HỆ THỐNG] Cài đặt thư viện hoàn tất.
) else (
    echo [HỆ THỐNG] Đang kích hoạt môi trường ảo có sẵn...
    call .venv\Scripts\activate
)

echo.
echo [HỆ THỐNG] Đang khởi chạy Máy chủ Web (Flask)...
echo [HỆ THỐNG] Trình duyệt sẽ tự động mở liên kết http://127.0.0.1:5000 sau vài giây...
echo.

:: Mở trình duyệt mặc định
start "" "http://127.0.0.1:5000"

:: Chạy ứng dụng Flask
python src/app.py

pause
