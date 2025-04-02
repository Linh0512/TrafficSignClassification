import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix
import cv2
import torch
import yaml
from tqdm import tqdm
import sys

# Thêm thư mục gốc vào sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, root_dir)

def parse_args():
    """Xử lý các đối số dòng lệnh"""
    parser = argparse.ArgumentParser(description='Đánh giá mô hình YOLOv8 cho nhận diện biển báo giao thông')
    parser.add_argument('--model', type=str, required=True, help='Đường dẫn đến model đã huấn luyện (*.pt)')
    parser.add_argument('--data', type=str, default=os.path.join(root_dir, 'data/dataset/data.yaml'), help='File dữ liệu yaml')
    parser.add_argument('--conf', type=float, default=0.25, help='Ngưỡng tin cậy')
    parser.add_argument('--iou', type=float, default=0.7, help='Ngưỡng IoU cho NMS')
    parser.add_argument('--img-size', type=int, default=640, help='Kích thước ảnh đầu vào')
    parser.add_argument('--batch-size', type=int, default=16, help='Kích thước batch')
    parser.add_argument('--device', type=str, default='', help='cuda device (0, 1, ...) hoặc cpu')
    parser.add_argument('--save-txt', action='store_true', help='Lưu kết quả nhận diện dưới dạng văn bản')
    parser.add_argument('--save-conf', action='store_true', help='Lưu ngưỡng tin cậy cùng với dự đoán')
    parser.add_argument('--visualize', action='store_true', help='Hiển thị một số kết quả nhận diện')
    parser.add_argument('--num-samples', type=int, default=10, help='Số mẫu để hiển thị khi visualize=True')
    parser.add_argument('--output-dir', type=str, default=os.path.join(root_dir, 'results', 'eval_results'), 
                      help='Thư mục lưu kết quả đánh giá')
    
    return parser.parse_args()

def load_class_names(data_yaml):
    """Tải tên các lớp từ file YAML"""
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    class_names = data['names']
    return class_names

def validate_model(args):
    """Đánh giá mô hình trên tập validation"""
    # Tải model
    model = YOLO(args.model)
    
    # Tải thông tin các class
    class_names = load_class_names(args.data)
    
    # Kiểm tra device
    device = args.device
    if device == '':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print(f"Đánh giá mô hình trên device: {device}")
    
    # Đánh giá model trên tập validation
    metrics = model.val(
        data=args.data,
        imgsz=args.img_size,
        batch=args.batch_size,
        conf=args.conf,
        iou=args.iou,
        device=device,
        verbose=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
    )
    
    print("\nKết quả đánh giá:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p:.4f}")
    print(f"Recall: {metrics.box.r:.4f}")
    
    return metrics

def plot_confusion_matrix(metrics, class_names, save_path='confusion_matrix.png'):
    """Vẽ ma trận nhầm lẫn từ kết quả đánh giá"""
    if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix is not None:
        cm = metrics.confusion_matrix
        
        # Chuẩn hóa confusion matrix
        cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
        
        # Tạo dataframe cho ma trận nhầm lẫn
        num_classes = len(class_names)
        if num_classes > cm_norm.shape[0]:
            num_classes = cm_norm.shape[0]
        
        cm_df = pd.DataFrame(cm_norm[:num_classes, :num_classes], 
                            index=class_names[:num_classes], 
                            columns=class_names[:num_classes])
        
        # Vẽ heatmap
        plt.figure(figsize=(16, 14))
        sns.heatmap(cm_df, annot=False, cmap='Blues', vmin=0, vmax=1)
        plt.title('Confusion Matrix (Normalized)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Đã lưu ma trận nhầm lẫn vào {save_path}")
    else:
        print("Không có dữ liệu ma trận nhầm lẫn")

def visualize_predictions(args, num_samples):
    """Hiển thị một số kết quả nhận diện từ model"""
    if not args.visualize:
        return
    
    # Tải model
    model = YOLO(args.model)
    
    # Tải thông tin các class
    class_names = load_class_names(args.data)
    
    # Tải config
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Lấy đường dẫn thư mục test
    test_dir = data_config.get('test', None)
    if test_dir is None:
        test_dir = data_config.get('val', None)
    
    if test_dir is None:
        print("Không tìm thấy đường dẫn thư mục test hoặc validation trong file config")
        return
    
    # Đảm bảo đường dẫn đúng
    if test_dir.startswith("../"):
        test_dir = os.path.join("dataset", test_dir[3:])
    
    # Lấy danh sách các file ảnh
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        print(f"Không tìm thấy ảnh trong {test_dir}")
        return
    
    # Chọn ngẫu nhiên một số ảnh để hiển thị
    selected_files = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
    
    # Tạo thư mục để lưu kết quả
    results_dir = 'eval_results'
    os.makedirs(results_dir, exist_ok=True)
    
    plt.figure(figsize=(20, 16))
    for i, img_file in enumerate(selected_files):
        # Đường dẫn ảnh
        img_path = os.path.join(test_dir, img_file)
        
        # Dự đoán với model
        results = model(img_path, conf=args.conf, iou=args.iou, verbose=False)[0]
        
        # Đọc ảnh để hiển thị
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Lấy các bounding box, confidence, và class
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        # Vẽ bounding box lên ảnh
        for box, conf, class_id in zip(boxes, confs, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            class_name = class_names[class_id]
            
            # Vẽ bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Thêm label
            label = f"{class_name} {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Hiển thị ảnh
        plt.subplot(np.ceil(num_samples / 2).astype(int), 2, i + 1)
        plt.imshow(image)
        plt.title(f"Sample {i+1}")
        plt.axis('off')
        
        # Lưu ảnh với kết quả nhận diện
        output_path = os.path.join(results_dir, f"pred_{img_file}")
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'predictions.png'))
    plt.close()
    
    print(f"Đã lưu mẫu kết quả nhận diện vào {results_dir}/predictions.png")

def plot_per_class_metrics(metrics, class_names, save_path='per_class_metrics.png'):
    """Vẽ biểu đồ các metrics theo từng lớp"""
    if not hasattr(metrics, 'per_class') or metrics.per_class is None:
        print("Không có dữ liệu metrics theo từng lớp")
        return
    
    # Trích xuất các metrics theo lớp
    precision = metrics.per_class['precision']
    recall = metrics.per_class['recall']
    f1 = 2 * (precision * recall) / (precision + recall + 1e-16)
    
    # Tạo DataFrame
    df = pd.DataFrame({
        'Class': class_names[:len(precision)],
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    # Sắp xếp theo F1-score
    df = df.sort_values('F1-Score', ascending=False)
    
    # Vẽ biểu đồ
    plt.figure(figsize=(15, 10))
    
    # Chỉ hiển thị top 20 class để dễ nhìn
    top_df = df.head(20)
    
    x = np.arange(len(top_df))
    width = 0.25
    
    plt.bar(x - width, top_df['Precision'], width=width, label='Precision')
    plt.bar(x, top_df['Recall'], width=width, label='Recall')
    plt.bar(x + width, top_df['F1-Score'], width=width, label='F1-Score')
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Per-class Evaluation Metrics (Top 20)')
    plt.xticks(x, top_df['Class'], rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Đã lưu biểu đồ metrics theo từng lớp vào {save_path}")
    
    # Lưu bảng metrics đầy đủ vào CSV
    df.to_csv('eval_results/per_class_metrics.csv', index=False)
    print("Đã lưu metrics theo từng lớp vào eval_results/per_class_metrics.csv")

def main():
    """Hàm chính"""
    args = parse_args()
    
    # Tạo thư mục lưu kết quả đánh giá
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Đánh giá model
    metrics = validate_model(args)
    
    # Tải tên các lớp
    class_names = load_class_names(args.data)
    
    # Vẽ ma trận nhầm lẫn
    plot_confusion_matrix(metrics, class_names, save_path=os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # Vẽ biểu đồ metrics theo từng lớp
    plot_per_class_metrics(metrics, class_names, save_path=os.path.join(args.output_dir, 'per_class_metrics.png'))
    
    # Hiển thị một số kết quả nhận diện
    if args.visualize:
        visualize_predictions(args, args.num_samples)
    
    print("\nĐánh giá mô hình hoàn tất!")

if __name__ == "__main__":
    main() 