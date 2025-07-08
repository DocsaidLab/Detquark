#!/usr/bin/env bash

set -e  # 若任一指令失敗，立即退出
set -u  # 未定義變數會被視為錯誤
set -o pipefail

# --- 檢查與下載子專案 ---
check_and_clone() {
    local dir=$1
    local repo=$2

    if [ ! -d "$dir" ]; then
        echo "[INFO] $dir not found. Cloning from $repo..."
        git clone "$repo" "$dir" || {
            echo "[ERROR] Failed to clone $repo"
            exit 1
        }
    else
        echo "[INFO] $dir already exists. Skipping clone."
    fi
}

check_and_clone "Capybara" "git@github.com:DocsaidLab/Capybara.git"
check_and_clone "Chameleon" "git@github.com:DocsaidLab/Chameleon.git"

# --- 建立 Docker 映像 ---
echo "[INFO] Building Docker image: detquark_train"
docker build \
    -f docker/Dockerfile \
    -t detquark_train .

echo "[INFO] Docker image built successfully."
