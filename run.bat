@echo off
title SMS Spam Detection System - CNN + SVM

echo ==========================================================
echo    HE THONG PHAT HIEN SMS SPAM - MO HINH LAI CNN + SVM
echo ==========================================================
echo.

:: Kiem tra su ton tai cua thu muc .venv
if not exist .venv (
    echo [SYSTEM] Khong tim thay moi truong ao .venv. Tien hanh khoi tao...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Tao moi truong ao that bai. Hay chac chan rang ban da cai dat Python 3.8+.
        pause
        exit /b 1
    )
    
    echo [SYSTEM] Khoi tao moi truong ao thanh cong.
    echo [SYSTEM] Dang kich hoat moi truong ao va cai dat thu vien...
    call .venv\Scripts\activate
    
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Cai dat cac thu vien that bai.
        pause
        exit /b 1
    )
    echo [SYSTEM] Cai dat thu vien hoan tat.
) else (
    echo [SYSTEM] Dang kich hoat moi truong ao co san...
    call .venv\Scripts\activate
)

echo.
echo [SYSTEM] Dang khoi chay May chu Web (Flask)...
echo [SYSTEM] Trinh duyet se tu dong mo lien ket http://127.0.0.1:5000 sau vai giay...
echo.

:: Mo trinh duyet mac dinh
start "" "http://127.0.0.1:5000"

:: Chay ung dung Flask
python src/app.py

pause
