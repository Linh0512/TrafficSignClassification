import os
import sys
import argparse
# Thêm thư mục gốc vào sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, root_dir)

from train import plot_training_results

def parse_args():
    """Xử lý các đối số dòng lệnh"""
    parser = argparse.ArgumentParser(description='Vẽ lại biểu đồ từ kết quả huấn luyện')
    parser.add_argument('--result-path', type=str, default=os.path.join(root_dir, 'runs/train/traffic_sign_detection'), 
                        help='Đường dẫn đến thư mục kết quả huấn luyện')
    
    return parser.parse_args()

def main():
    """Hàm chính"""
    args = parse_args()
    
    # Kiểm tra thư mục kết quả có tồn tại không
    if not os.path.exists(args.result_path):
        print(f"Lỗi: Không tìm thấy thư mục kết quả tại {args.result_path}")
        sys.exit(1)
    
    # Kiểm tra file CSV có tồn tại không
    csv_path = os.path.join(args.result_path, "results.csv")
    if not os.path.exists(csv_path):
        print(f"Lỗi: Không tìm thấy file CSV kết quả tại {csv_path}")
        sys.exit(1)
    
    print(f"Đang vẽ lại biểu đồ từ kết quả trong thư mục: {args.result_path}")
    
    # Gọi hàm vẽ biểu đồ từ file train.py
    plot_training_results(args.result_path)
    
    print("Hoàn tất!")

if __name__ == "__main__":
    main() 