#!/bin/bash

echo "======================================"
echo " DarkPaw WebServer 시작"
echo "======================================"

echo "[1/3] 기존 서버 종료"

sudo pkill -f server.py 2>/dev/null
sudo pkill -f webServer.py 2>/dev/null

sleep 1

echo "[2/3] 서버 디렉토리 이동"

cd ~/knu_adeept_darkpaw/server || {
    echo "ERROR: 서버 폴더를 찾을 수 없습니다."
    exit 1
}

echo "[3/3] WebServer 실행"
echo "--------------------------------------"

sudo python3 webServer.py