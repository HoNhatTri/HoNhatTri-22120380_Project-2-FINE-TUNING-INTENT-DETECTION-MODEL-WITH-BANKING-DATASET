#!/bin/bash

# Cài đặt các thư viện cần thiết
pip install unsloth
pip install --no-deps trl peft accelerate bitsandbytes
pip install datasets pyyaml

# Chạy script huấn luyện
python scripts/train.py