# Banking Intent Classification with Llama 3.2 & Unsloth

Bài tập Tinh chỉnh mô hình Ngôn ngữ lớn Llama-3.2-1B để thực hiện bài toán Phân loại ý định khách hàng trong lĩnh vực ngân hàng. 

Bài tập được thực hiện phục vụ cho học phần **Ứng dụng Xử lý ngôn ngữ tự nhiên trong doanh nghiệp**.

---

## Hướng dẫn triển khai trên Google Colab

Để hệ thống hoạt động trơn tru, vui lòng thực hiện tuần tự 4 bước dưới đây:

### Bước 1: Cài đặt môi trường
Dự án yêu cầu Python 3.10+ và môi trường có hỗ trợ GPU (NVIDIA CUDA). Để cài đặt tất cả các thư viện cần thiết, hãy chạy lệnh:
```bash
pip install -r requirements.txt
```

### Bước 2: Tải và Chuẩn bị dữ liệu
Dự án sử dụng tập dữ liệu **BANKING77**. Thực hiện tiền xử lý dữ liệu, hãy chạy lệnh:
```bash
python scripts/preprocess_data.py
```
Lệnh sẽ thực hiện tải tập dữ liệu gốc (train: 10003, test: 3080), chia tập con từ tập gốc (train: 3000, test: 500) để làm tập dữ liệu thực hiện bài tập, đồng thời làm sạch dữ liệu trước khi bước vào quá trình huấn luyện.

### Bước 3: Huấn luyện mô hình
Bắt đầu quá trình tinh chỉnh mô hình bằng kỹ thuật **LoRA** (Low-Rank Adaptation) và huấn luyện mô hình, chạy lệnh:
```bash
!chmod +x train.sh ## mở quyền truy cập
./train.sh
```

### Bước 4: Chạy mô hình
Sau khi huấn luyện xong, chạy lệnh sau để thực hiện ví dụ dự đoán ý định khách hàng từ các câu hỏi ngẫu nhiên từ tập test, đồng thời tính toán độ chính xác của mô hình trên tập test:
```bash
!chmod +x inference.sh ## mở quyền truy cập
!./inference.sh
```

## Cấu trúc thư mục

```text
├── configs/
│   ├── train.yaml          # Cấu hình siêu tham số
│   └── inference.yaml      # Cấu hình đường dẫn cho suy luận
├── scripts/
│   ├── preprocess_data.py  # Mã nguồn tiền xử lý dữ liệu thô
│   ├── train.py            # Mã nguồn cấu hình dữ liệu và huấn luyện
│   └── inference.py        # Mã nguồn chạy mô hình và tính Accuracy
├── sample_data/
│   ├── train.csv           # Dữ liệu huấn luyện (Sinh ra từ file preprocess_data.py)
│   └── test.csv            # Dữ liệu đánh giá (Sinh ra từ file preprocess_data.py)
├── saved_models/
│   └── banking_intent_model/ # Thư mục lưu mô hình sau khi train
├── requirements.txt        # Danh sách thư viện Python
├── train.sh                # Script tự động chạy train
└── inference.sh            # Script tự động chạy inference
```

## Link video demo
[Video_Demo](https://drive.google.com/file/d/1_ZAQO0a0jw6FUOjXtIfjuFWbn2-gH6IM/view?usp=drive_link)




