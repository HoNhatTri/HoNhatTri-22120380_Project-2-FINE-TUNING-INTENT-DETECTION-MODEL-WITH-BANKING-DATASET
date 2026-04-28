#!/bin/bash

# Cài đặt các thư viện cần thiết
pip install unsloth
pip install --no-deps peft accelerate bitsandbytes
pip install pandas pyyaml tqdm

# Chạy script suy luận và đánh giá Accuracy
python scripts/inference.py