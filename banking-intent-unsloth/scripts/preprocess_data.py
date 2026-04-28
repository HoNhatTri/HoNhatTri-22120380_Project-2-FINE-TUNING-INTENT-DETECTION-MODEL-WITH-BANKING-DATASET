import os
import pandas as pd
from datasets import load_dataset

def main():
    print("=== TẢI DỮ LIỆU ===")
    print("Đang tải tập dữ liệu BANKING77 từ Hugging Face...")
    dataset = load_dataset("banking77")
    
    df_train = dataset['train'].to_pandas()
    df_test = dataset['test'].to_pandas()
    
    print(f"Kích thước dataset -> Train: {df_train.shape[0]} câu, Test: {df_test.shape[0]} câu.")

    print("\n=== LẤY MẪU ===")
    SAMPLE_TRAIN_SIZE = 3000
    SAMPLE_TEST_SIZE = 500
    
    sampled_train = df_train.sample(n=SAMPLE_TRAIN_SIZE, random_state=42)
    sampled_test = df_test.sample(n=SAMPLE_TEST_SIZE, random_state=42)
    
    print(f"Kích thước tập Subset -> Train: {sampled_train.shape[0]}, Test: {sampled_test.shape[0]}.")

    print("\n=== KIỂM TRA & LÀM SẠCH DỮ LIỆU ===")
    # Loại bỏ dữ liệu rỗng
    null_train = sampled_train[sampled_train[['text', 'label']].isnull().any(axis=1)]
    null_test = sampled_test[sampled_test[['text', 'label']].isnull().any(axis=1)]    
    print(f"Số câu bị Null trong tập Train: {len(null_train)}")
    print(f"Số câu bị Null trong tập Test: {len(null_test)}")

    sampled_train.dropna(subset=['text', 'label'], inplace=True)
    sampled_test.dropna(subset=['text', 'label'], inplace=True)
    
    # Xóa các câu bị trùng lặp
    dup_train = sampled_train[sampled_train.duplicated(subset=['text'], keep='first')]
    dup_test = sampled_test[sampled_test.duplicated(subset=['text'], keep='first')]
    print(f"Số câu bị trùng lặp trong tập Train: {len(dup_train)}")
    print(f"Số câu bị trùng lặp trong tập Test: {len(dup_test)}")

    sampled_train.drop_duplicates(subset=['text'], keep='first', inplace=True)
    sampled_test.drop_duplicates(subset=['text'], keep='first', inplace=True)
    
    # Lọc nhiễu: Loại bỏ các câu quá ngắn (< 3 từ) hoặc quá dài (> 100 từ)
    sampled_train['word_count'] = sampled_train['text'].apply(lambda x: len(str(x).split()))
    sampled_test['word_count'] = sampled_test['text'].apply(lambda x: len(str(x).split()))

    noisy_train = sampled_train[(sampled_train['word_count'] < 3) | (sampled_train['word_count'] > 100)]
    noisy_test = sampled_test[(sampled_test['word_count'] < 3) | (sampled_test['word_count'] > 100)]

    print(f"Số lượng câu nhiễu trong tập Train: {len(noisy_train)}")
    print(f"Số lượng câu nhiễu trong tập Test: {len(noisy_test)}")

    sampled_train = sampled_train[(sampled_train['word_count'] >= 3) & (sampled_train['word_count'] <= 100)]
    sampled_test = sampled_test[(sampled_test['word_count'] >= 3) & (sampled_test['word_count'] <= 100)]

    sampled_train.drop(columns=['word_count'], inplace=True)
    sampled_test.drop(columns=['word_count'], inplace=True)

    print(f"Sau khi làm sạch -> Train còn: {sampled_train.shape[0]} câu, Test còn: {sampled_test.shape[0]} câu.")


    print("\n=== LƯU DỮ LIỆU ===")
    os.makedirs("sample_data", exist_ok=True)
    
    sampled_train.to_csv("sample_data/train.csv", index=False)
    sampled_test.to_csv("sample_data/test.csv", index=False)
    
    print("Đã lưu 'train.csv' và 'test.csv' vào thư mục 'sample_data/'.")

if __name__ == "__main__":
    main()