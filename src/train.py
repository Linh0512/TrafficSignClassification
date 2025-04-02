import os
import yaml
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch
import sys

# Thêm thư mục gốc vào sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, root_dir)

def parse_args():
    """Xử lý các đối số dòng lệnh"""
    parser = argparse.ArgumentParser(description='Huấn luyện mô hình YOLOv8 cho nhận diện biển báo giao thông')
    parser.add_argument('--model', type=str, default=os.path.join(root_dir, 'models', 'yolov8n.pt'), 
                      help='Mô hình YOLOv8 muốn sử dụng (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)')
    parser.add_argument('--epochs', type=int, default=50, help='Số epochs huấn luyện')
    parser.add_argument('--batch-size', type=int, default=16, help='Kích thước batch')
    parser.add_argument('--img-size', type=int, default=640, help='Kích thước ảnh đầu vào')
    parser.add_argument('--device', type=str, default='', help='cuda device (0, 1, ...) hoặc cpu')
    parser.add_argument('--project', type=str, default=os.path.join(root_dir, 'runs/train'), help='Thư mục lưu kết quả')
    parser.add_argument('--name', type=str, default='traffic_sign_detection', help='Tên thử nghiệm')
    parser.add_argument('--data', type=str, default=os.path.join(root_dir, 'data/dataset/data.yaml'), 
                      help='File dữ liệu yaml')
    parser.add_argument('--patience', type=int, default=10, help='Số epochs đợi trước khi early stopping')
    
    return parser.parse_args()

def plot_training_results(results_path):
    """Vẽ biểu đồ từ kết quả huấn luyện"""
    if os.path.exists(f"{results_path}/results.csv"):
        # Đọc kết quả từ file CSV
        import pandas as pd
        results = pd.read_csv(f"{results_path}/results.csv")
        
        # In các cột có sẵn trong file CSV để kiểm tra
        print("Các cột có trong file results.csv:")
        print(results.columns.tolist())
        
        # Vẽ biểu đồ mAP
        plt.figure(figsize=(12, 10))
        
        # Vẽ mAP50 nếu có
        if 'metrics/mAP50(B)' in results.columns:
            plt.subplot(2, 2, 1)
            plt.plot(results['epoch'], results['metrics/mAP50(B)'], label='validation mAP50')
            plt.xlabel('Epoch')
            plt.ylabel('mAP50')
            plt.title('Validation mAP50')
            plt.legend()
        
        # Vẽ mAP50-95 nếu có
        if 'metrics/mAP50-95(B)' in results.columns:
            plt.subplot(2, 2, 2)
            plt.plot(results['epoch'], results['metrics/mAP50-95(B)'], label='validation mAP50-95')
            plt.xlabel('Epoch')
            plt.ylabel('mAP50-95')
            plt.title('Validation mAP50-95')
            plt.legend()
        
        # Tìm tất cả các cột loss
        loss_columns = [col for col in results.columns if 'loss' in col.lower()]
        print("Các cột liên quan đến loss:", loss_columns)
        
        # Vẽ loss
        plt.subplot(2, 2, 3)
        
        # Vẽ tất cả các cột loss được tìm thấy
        for col in loss_columns:
            plt.plot(results['epoch'], results[col], label=col)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        
        # Vẽ các metric khác (nếu có)
        plt.subplot(2, 2, 4)
        
        # Tìm các cột metrics
        metric_columns = [col for col in results.columns if 'metrics' in col.lower() and 'mAP' not in col]
        if not metric_columns:
            # Nếu không có các cột metrics khác, vẽ lại mAP
            if 'metrics/mAP50(B)' in results.columns and 'metrics/mAP50-95(B)' in results.columns:
                plt.plot(results['epoch'], results['metrics/mAP50(B)'], label='mAP50')
                plt.plot(results['epoch'], results['metrics/mAP50-95(B)'], label='mAP50-95')
                plt.title('mAP Metrics')
        else:
            for col in metric_columns:
                plt.plot(results['epoch'], results[col], label=col)
            plt.title('Other Metrics')
        
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{results_path}/training_plots.png")
        plt.close()
        
        print(f"Đã lưu biểu đồ huấn luyện vào {results_path}/training_plots.png")
    else:
        print(f"Không tìm thấy file kết quả tại {results_path}/results.csv")

def check_dataset(data_yaml):
    """Kiểm tra và xác nhận dataset"""
    try:
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        # Kiểm tra các trường bắt buộc
        required_fields = ['train', 'val', 'test', 'nc', 'names']
        for field in required_fields:
            if field not in data:
                print(f"Lỗi: Thiếu trường '{field}' trong file {data_yaml}")
                return False
        
        # Kiểm tra số lượng tên lớp
        if len(data['names']) != data['nc']:
            print(f"Cảnh báo: Số lượng tên lớp ({len(data['names'])}) không khớp với nc ({data['nc']})")
        
        print(f"Dataset hợp lệ với {data['nc']} lớp.")
        print(f"Đường dẫn huấn luyện: {data['train']}")
        print(f"Đường dẫn validation: {data['val']}")
        print(f"Đường dẫn kiểm thử: {data['test']}")
        return True
    
    except Exception as e:
        print(f"Lỗi khi đọc file dataset: {e}")
        return False

def train(args):
    """Huấn luyện mô hình YOLOv8"""
    # Kiểm tra dataset trước khi huấn luyện
    if not check_dataset(args.data):
        print("Dừng huấn luyện do dataset không hợp lệ")
        return
    
    # Kiểm tra GPU
    device = args.device
    if device == '':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    if device != 'cpu':
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"Đã phát hiện {gpu_count} GPU.")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"GPU {i}: {gpu_name}")
        else:
            print("Không phát hiện GPU. Sử dụng CPU.")
            device = 'cpu'
    
    print(f"Sử dụng device: {device}")
    
    # Tạo mô hình
    model = YOLO(args.model)
    
    # Cấu hình huấn luyện
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        patience=args.patience,
        device=device,
        project=args.project,
        name=args.name,
        verbose=True,
        exist_ok=True,
        pretrained=True,
        optimizer='SGD',  # 'SGD', 'Adam', or 'AdamW'
        lr0=1e-3,  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
        lrf=0.01,  # final learning rate (lr0 * lrf)
        momentum=0.937,  # SGD momentum/Adam beta1
        weight_decay=0.0005,  # optimizer weight decay 5e-4
        warmup_epochs=3.0,  # warmup epochs (fractions ok)
        warmup_momentum=0.8,  # warmup initial momentum
        warmup_bias_lr=0.1,  # warmup initial bias lr
        close_mosaic=10,  # disable mosaic augmentation for final 10 epochs
        box=7.5,  # box loss gain
        cls=0.5,  # cls loss gain
        dfl=1.5,  # dfl loss gain
        amp=True,  # Automatic Mixed Precision
        plots=True,  # save plots
        save=True,  # save checkpoints
    )
    
    # Vẽ và lưu biểu đồ kết quả huấn luyện
    plot_training_results(f"{args.project}/{args.name}")
    
    # Hiển thị kết quả
    print("\nKết quả huấn luyện:")
    print(f"Best mAP50: {results.best_fitness:.4f}")
    print(f"Đã lưu model tốt nhất vào: {results.best}")
    
    return results

def main():
    """Hàm chính"""
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main() 