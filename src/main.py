import os
import sys
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Thêm thư mục gốc vào sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, root_dir)

def parse_args():
    """Xử lý các đối số dòng lệnh"""
    parser = argparse.ArgumentParser(description="Chương trình chính cho nhận diện biển báo giao thông với YOLOv12")
    parser.add_argument("--mode", type=str, choices=["train", "predict", "evaluate", "visualize", "download", "eda"], 
                        required=True, help="Chế độ chạy: train, predict, evaluate, visualize, download, eda")
    parser.add_argument("--model", type=str, help="Đường dẫn đến model YOLOv12")
    parser.add_argument("--data", type=str, default=os.path.join(root_dir, "data/data.yaml"), 
                        help="Đường dẫn đến file data.yaml")
    parser.add_argument("--source", type=str, help="Đường dẫn đến ảnh/thư mục/video cần dự đoán")
    parser.add_argument("--epochs", type=int, default=50, help="Số epochs huấn luyện")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--img_size", type=int, default=640, help="Kích thước ảnh")
    parser.add_argument("--device", type=str, default="", help="CUDA device, vd: 0, 1, 2, 3 hoặc cpu")
    parser.add_argument("--workers", type=int, default=8, help="Số worker thread")
    parser.add_argument("--conf_thres", type=float, default=0.25, help="Ngưỡng tin cậy")
    parser.add_argument("--iou_thres", type=float, default=0.45, help="Ngưỡng IoU cho NMS")
    parser.add_argument("--project", type=str, default=os.path.join(root_dir, "outputs/runs"), help="Thư mục lưu kết quả")
    parser.add_argument("--name", type=str, default="exp", help="Tên thư mục con lưu kết quả")
    parser.add_argument("--save_txt", action="store_true", help="Lưu kết quả dạng .txt")
    parser.add_argument("--save_conf", action="store_true", help="Lưu điểm tin cậy với kết quả")
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    
    # Chọn chế độ chạy
    if args.mode == "train":
        from src.train import train
        train(args)
    elif args.mode == "predict":
        from src.predict import predict
        predict(args)
    elif args.mode == "evaluate":
        from src.evaluate import evaluate
        evaluate(args)
    elif args.mode == "visualize":
        from src.visualize_labels import visualize_labels
        visualize_labels(args)
    elif args.mode == "download":
        from src.download_dataset import download_dataset
        download_dataset(args)
    elif args.mode == "eda":
        from src.eda import exploratory_data_analysis
        exploratory_data_analysis(args)
    
if __name__ == "__main__":
    main() 