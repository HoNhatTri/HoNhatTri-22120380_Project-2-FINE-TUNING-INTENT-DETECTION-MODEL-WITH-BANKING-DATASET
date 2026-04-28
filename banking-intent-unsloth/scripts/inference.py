import yaml
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel

class IntentClassification:
    def __init__(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        checkpoint_dir = self.config["model_path"]
        max_seq_length = self.config.get("max_seq_length", 256)
        
        print(f"--- Đang tải mô hình từ: {checkpoint_dir} ---")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = checkpoint_dir,
            max_seq_length = max_seq_length,
            load_in_4bit = True,
        )
        
        FastLanguageModel.for_inference(self.model)

        self.label_map = {
            0: "activate_my_card", 1: "age_limit", 2: "apple_pay_or_google_pay", 3: "atm_support",
            4: "automatic_top_up", 5: "balance_not_updated_after_bank_transfer", 
            6: "balance_not_updated_after_cheque_or_cash_deposit", 7: "beneficiary_not_allowed",
            8: "cancel_transfer", 9: "card_about_to_expire", 10: "card_acceptance", 11: "card_arrival",
            12: "card_delivery_estimate", 13: "card_linking", 14: "card_not_working", 15: "card_payment_fee_charged",
            16: "card_payment_not_recognised", 17: "card_payment_wrong_exchange_rate", 18: "card_swallowed",
            19: "cash_withdrawal_charge", 20: "cash_withdrawal_not_recognised", 21: "change_pin",
            22: "compromised_card", 23: "contactless_not_working", 24: "country_support", 
            25: "declined_card_payment", 26: "declined_cash_withdrawal", 27: "declined_transfer",
            28: "direct_debit_payment_not_recognised", 29: "disposable_card_limits", 30: "edit_personal_details",
            31: "exchange_charge", 32: "exchange_rate", 33: "exchange_via_app", 34: "extra_charge_on_statement",
            35: "failed_transfer", 36: "fiat_currency_support", 37: "get_disposable_virtual_card",
            38: "get_physical_card", 39: "getting_spare_card", 40: "getting_virtual_card", 
            41: "lost_or_stolen_card", 42: "lost_or_stolen_phone", 43: "order_physical_card",
            44: "passcode_forgotten", 45: "pending_card_payment", 46: "pending_cash_withdrawal",
            47: "pending_top_up", 48: "pending_transfer", 49: "pin_blocked", 50: "receiving_money",
            51: "Refund_not_showing_up", 52: "request_refund", 53: "reverted_card_payment?",
            54: "supported_cards_and_currencies", 55: "terminate_account", 56: "top_up_by_bank_transfer_charge",
            57: "top_up_by_card_charge", 58: "top_up_by_cash_or_cheque", 59: "top_up_failed",
            60: "top_up_limits", 61: "top_up_reverted", 62: "topping_up_by_card", 63: "transaction_charged_twice",
            64: "transfer_fee_charged", 65: "transfer_into_account", 66: "transfer_not_received_by_recipient",
            67: "transfer_timing", 68: "unable_to_verify_identity", 69: "verify_my_identity",
            70: "verify_source_of_funds", 71: "verify_top_up", 72: "virtual_card_not_working",
            73: "visa_or_mastercard", 74: "why_verify_identity", 75: "wrong_amount_of_cash_received",
            76: "wrong_exchange_rate_for_cash_withdrawal"
        }
        
        self.prompt_template = "Intent classification. Input: {}\nLabel: "

    def __call__(self, message):

        input_text = self.prompt_template.format(message)
        
        inputs = self.tokenizer([input_text], return_tensors="pt").to("cuda")
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=64,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        predicted_label = decoded_output.split("Label: ")[-1].strip()
        
        try:
            label_id = int(predicted_label)
            return self.label_map.get(label_id, f"Unknown Label ({label_id})")
        except ValueError:
            return predicted_label


if __name__ == "__main__":
    classifier = IntentClassification("configs/inference.yaml")

    test_path = classifier.config["test_path"]
    df_test = pd.read_csv(test_path)
    
    try:
        test_path = classifier.config["test_path"]
        df_test = pd.read_csv(test_path)
        
        correct = 0
        total = len(df_test)

        for _, row in tqdm(df_test.iterrows(), total=total):
            message = row['text']
            
            true_label_id = int(row['label'])
            true_label_name = classifier.label_map.get(true_label_id, str(true_label_id))  
            
            pred_label_name = classifier(message)
            
            
            if pred_label_name.strip().lower() == true_label_name.strip().lower():
                correct += 1

        
        print("\n[VÍ DỤ DỰ ĐOÁN NGẪU NHIÊN TỪ TẬP TEST]")
        random_sample = df_test.sample(n=1).iloc[0]
        sample_text = random_sample['text']
        result = classifier(sample_text)
        true_label_id = int(random_sample['label'])
        true_label_name = classifier.label_map.get(true_label_id, str(true_label_id))

        print(f"Câu hỏi: {sample_text}")
        print(f"Dự đoán ý định: {result}")
        print(f"Nhãn thực tế: {true_label_name}")

        if result.lower() == true_label_name.lower():
            print("=> Dự đoán chính xác!")
        else:
            print("=> Dự đoán chưa chính xác!")
        print("-" * 50)

        print("\n[TÍNH TOÁN CHỈ SỐ ACCURACY]")
        accuracy = (correct / total) * 100
        print(f"Đánh giá mô hình trên tập Test:")
        print(f"   - Tổng số mẫu kiểm tra: {total}")
        print(f"   - Số mẫu dự đoán đúng: {correct}")
        print(f"   - Độ chính xác (Accuracy): {accuracy:.2f}%")
        
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file test.csv tại đường dẫn đã cấu hình.")