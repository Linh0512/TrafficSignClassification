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
    parser = argparse.ArgumentParser(description='Đánh giá mô hình YOLOv12 cho nhận diện biển báo giao thông')
    parser.add_argument('--model', type=str, required=True, help='Đường dẫn đến model đã huấn luyện (*.pt)')
    parser.add_argument('--data', type=str, default=os.path.join(root_dir, 'data/data.yaml'), help='File dữ liệu yaml')
    parser.add_argument('--conf', type=float, default=0.25, help='Ngưỡng tin cậy')
    parser.add_argument('--iou', type=float, default=0.7, help='Ngưỡng IoU cho NMS')
    parser.add_argument('--img-size', type=int, default=640, help='Kích thước ảnh đầu vào')
    parser.add_argument('--batch-size', type=int, default=16, help='Kích thước batch')
    parser.add_argument('--device', type=str, default='', help='cuda device (0, 1, ...) hoặc cpu')
    parser.add_argument('--save-txt', action='store_true', help='Lưu kết quả nhận diện dưới dạng văn bản')
    parser.add_argument('--save-conf', action='store_true', help='Lưu ngưỡng tin cậy cùng với dự đoán')
    parser.add_argument('--visualize', action='store_true', help='Hiển thị một số kết quả nhận diện')
    parser.add_argument('--num-samples', type=int, default=10, help='Số mẫu để hiển thị khi visualize=True')
    parser.add_argument('--output', type=str, default=os.path.join(root_dir, 'outputs/results/evaluation'), help='Thư mục lưu kết quả đánh giá')
    
    args = parser.parse_args()
    return args

def load_classes(data_yaml):
    """Tải danh sách lớp từ file yaml"""
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['names']

def evaluate(args):
    """Đánh giá mô hình YOLOv12"""
    # Kiểm tra đầu vào
    if not os.path.exists(args.model):
        print(f"Lỗi: Không tìm thấy model tại {args.model}")
        return
    
    # Xử lý output path - sử dụng project nếu không có output
    if hasattr(args, 'output') and args.output:
        output_dir = args.output
    elif hasattr(args, 'project') and args.project:
        output_dir = os.path.join(args.project, 'evaluation')
    else:
        output_dir = os.path.join(root_dir, 'outputs/runs/evaluation')
    
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    # Tải classes
    class_names = load_classes(args.data)
    print(f"Đã tải {len(class_names)} classes từ {args.data}")
    
    # Thiết lập device
    device = args.device
    if device == '':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Sử dụng device: {device}")
    
    # Tải model
    try:
        model = YOLO(args.model)
        print(f"Đã tải model từ {args.model}")
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        return
    
    # Xử lý các tham số từ main.py
    conf = getattr(args, 'conf', getattr(args, 'conf_thres', 0.25))
    iou = getattr(args, 'iou', getattr(args, 'iou_thres', 0.7))
    img_size = getattr(args, 'img_size', getattr(args, 'img_size', 640))
    batch_size = getattr(args, 'batch_size', getattr(args, 'batch_size', 16))
    save_txt = getattr(args, 'save_txt', False)
    save_conf = getattr(args, 'save_conf', False)
    
    # Thực hiện đánh giá
    try:
        print(f"Đánh giá model trên tập dữ liệu {args.data}")
        metrics = model.val(
            data=args.data,
            conf=conf,
            iou=iou,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            save_txt=save_txt,
            save_conf=save_conf,
            save_json=True,  # Lưu kết quả dạng COCO format để phân tích
            project=output_dir,
            name='val',
            exist_ok=True,
            verbose=True
        )
        
        # In kết quả
        print("\nKết quả đánh giá:")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP75: {metrics.box.map75:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        
        # Phân tích thêm và trực quan kết quả
        visualize_results(model, args, class_names, output_dir, conf, iou)
        
        # Lưu kết quả thành báo cáo
        save_report(metrics, class_names, output_dir)
        
        return metrics
        
    except Exception as e:
        print(f"Lỗi khi đánh giá model: {e}")
        return None

# Alias để tương thích ngược
validate = evaluate

def visualize_results(model, args, class_names, output_dir, conf, iou):
    """Hiển thị và trực quan hóa một số kết quả"""
    if not hasattr(args, 'visualize') or not args.visualize:
        return
    
    # Tải dữ liệu từ file YAML
    with open(args.data, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    test_path = data['test']
    if not os.path.exists(test_path):
        print(f"Đường dẫn tập kiểm thử không tồn tại: {test_path}")
        return
    
    # Lấy đường dẫn thư mục chứa ảnh test
    test_images_path = os.path.join(test_path, 'images')
    if not os.path.exists(test_images_path):
        print(f"Không tìm thấy thư mục ảnh test: {test_images_path}")
        return
    
    # Lấy danh sách ảnh
    images = [os.path.join(test_images_path, f) for f in os.listdir(test_images_path) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not images:
        print("Không tìm thấy ảnh để hiển thị")
        return
    
    # Lấy mẫu ngẫu nhiên
    num_samples = min(args.num_samples, len(images))
    sample_indices = np.random.choice(len(images), num_samples, replace=False)
    
    # Dự đoán trên các mẫu ngẫu nhiên
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, 2)
    
    for i, idx in enumerate(sample_indices):
        image_path = images[idx]
        
        # Đọc ảnh gốc
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Dự đoán với model
        results = model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            device=args.device,
            verbose=False
        )[0]
        
        # Vẽ ảnh gốc
        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title(f"Ảnh gốc: {os.path.basename(image_path)}")
        axes[i, 0].axis("off")
        
        # Vẽ ảnh với kết quả dự đoán
        plot_img = results.plot()
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB)
        axes[i, 1].imshow(plot_img)
        axes[i, 1].set_title(f"Dự đoán")
        axes[i, 1].axis("off")
    
    plt.tight_layout()
    viz_path = os.path.join(output_dir, 'sample_predictions.png')
    plt.savefig(viz_path)
    plt.close()
    
    print(f"Đã lưu ảnh trực quan mẫu vào {viz_path}")

def save_report(metrics, class_names, output_dir):
    """Lưu báo cáo đánh giá"""
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("BÁO CÁO ĐÁNH GIÁ MÔ HÌNH YOLOv12\n")
        f.write("="*50 + "\n\n")
        
        f.write("Metrics tổng quát:\n")
        f.write(f"mAP50-95: {metrics.box.map:.4f}\n")
        f.write(f"mAP50: {metrics.box.map50:.4f}\n")
        f.write(f"mAP75: {metrics.box.map75:.4f}\n")
        f.write(f"Precision: {metrics.box.mp:.4f}\n")
        f.write(f"Recall: {metrics.box.mr:.4f}\n\n")
        
        f.write("Thông tin từng lớp:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"Class {i} ({class_name}):\n")
            # Kiểm tra xem metrics cho lớp này có tồn tại không
            if hasattr(metrics.by_class, 'ap50') and i < len(metrics.by_class.ap50):
                f.write(f"  - AP50: {metrics.by_class.ap50[i]:.4f}\n")
                f.write(f"  - Precision: {metrics.by_class.p[i]:.4f}\n")
                f.write(f"  - Recall: {metrics.by_class.r[i]:.4f}\n")
            else:
                f.write(f"  - Không có thông tin metrics cho lớp này\n")
        
        f.write("\nLưu ý: Kết quả có thể thay đổi tùy thuộc vào cấu hình và ngưỡng được sử dụng.\n")
    
    print(f"Đã lưu báo cáo đánh giá vào {report_path}")

def main():
    """Hàm chính"""
    args = parse_args()
    evaluate(args)

if __name__ == "__main__":
    main() 