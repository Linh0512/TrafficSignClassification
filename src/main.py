import os
import sys
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Traffic Sign Detection Project')
    
    # Các thao tác chính
    parser.add_argument('--action', type=str, required=True, choices=['eda', 'train', 'evaluate', 'predict', 'plot', 'visualize-labels', 'download-dataset'],
                      help='Hành động muốn thực hiện: eda, train, evaluate, predict, plot, visualize-labels, download-dataset')
    
    # Các tham số chung
    parser.add_argument('--model', type=str, default='models/yolov8n.pt',
                      help='Đường dẫn đến model (*.pt)')
    parser.add_argument('--data', type=str, default='data/dataset/data.yaml',
                      help='File dữ liệu yaml')
    
    # Các tham số cho huấn luyện
    parser.add_argument('--epochs', type=int, default=50,
                      help='Số epochs huấn luyện')
    parser.add_argument('--batch-size', type=int, default=16,
                      help='Kích thước batch')
    parser.add_argument('--img-size', type=int, default=640,
                      help='Kích thước ảnh đầu vào')
    parser.add_argument('--device', type=str, default='',
                      help='cuda device (0, 1, ...) hoặc cpu')
    
    # Các tham số cho predict
    parser.add_argument('--source', type=str, default='',
                      help='Đường dẫn đến ảnh hoặc thư mục chứa ảnh cần dự đoán')
    parser.add_argument('--conf', type=float, default=0.25,
                      help='Ngưỡng tin cậy')
    parser.add_argument('--iou', type=float, default=0.7,
                      help='Ngưỡng IoU cho NMS')
    parser.add_argument('--view-img', action='store_true',
                      help='Hiển thị ảnh kết quả')
    
    # Các tham số cho plot
    parser.add_argument('--result-path', type=str, default='runs/train/traffic_sign_detection',
                      help='Đường dẫn đến thư mục kết quả huấn luyện')
    
    # Các tham số cho evaluate
    parser.add_argument('--visualize', action='store_true',
                      help='Hiển thị một số kết quả nhận diện')
    parser.add_argument('--num-samples', type=int, default=10,
                      help='Số mẫu để hiển thị khi visualize=True')
    
    # Các tham số cho visualize labels
    parser.add_argument('--output', type=str, default='data/dataset/data_renamed.yaml',
                      help='Đường dẫn đến file data.yaml đầu ra với tên nhãn mới')
    parser.add_argument('--sample-per-class', type=int, default=5, 
                      help='Số lượng ảnh mẫu để hiển thị cho mỗi lớp')
    
    # Các tham số cho download dataset
    parser.add_argument('--api-key', type=str, default='',
                      help='Roboflow API key')
    
    return parser.parse_args()

def run_eda():
    print("Đang chạy phân tích dữ liệu thăm dò (EDA)...")
    os.system(f"python src/eda.py")

def run_train(args):
    print("Đang huấn luyện mô hình...")
    cmd = (f"python src/train.py --model {args.model} --epochs {args.epochs} "
           f"--batch-size {args.batch_size} --img-size {args.img_size} "
           f"--data {args.data}")
    
    if args.device:
        cmd += f" --device {args.device}"
    
    os.system(cmd)

def run_evaluate(args):
    print("Đang đánh giá mô hình...")
    cmd = (f"python src/evaluate.py --model {args.model} --data {args.data} "
           f"--conf {args.conf} --iou {args.iou}")
    
    if args.device:
        cmd += f" --device {args.device}"
    
    if args.visualize:
        cmd += f" --visualize --num-samples {args.num_samples}"
    
    os.system(cmd)

def run_predict(args):
    if not args.source:
        print("Lỗi: Thiếu đối số --source. Hãy chỉ định đường dẫn đến ảnh hoặc thư mục ảnh.")
        sys.exit(1)
        
    print("Đang thực hiện dự đoán...")
    cmd = (f"python src/predict.py --model {args.model} --source {args.source} "
           f"--conf {args.conf} --iou {args.iou} "
           f"--data {args.data}")
    
    if args.device:
        cmd += f" --device {args.device}"
    
    if args.view_img:
        cmd += " --view-img"
    
    os.system(cmd)

def run_plot(args):
    print("Đang vẽ biểu đồ...")
    os.system(f"python src/plot_results.py --result-path {args.result_path}")

def run_visualize_labels(args):
    print("Đang hiển thị và cho phép sửa đổi nhãn...")
    cmd = (f"python src/visualize_labels.py --data {args.data} "
           f"--output {args.output} --sample-per-class {args.sample_per_class}")
    os.system(cmd)

def run_download_dataset(args):
    print("Đang tải dữ liệu từ Roboflow...")
    cmd = f"python src/download_dataset.py --data {args.data}"
    
    if args.api_key:
        cmd += f" --api-key {args.api_key}"
    
    os.system(cmd)

def main():
    args = parse_args()
    
    if args.action == 'eda':
        run_eda()
    elif args.action == 'train':
        run_train(args)
    elif args.action == 'evaluate':
        run_evaluate(args)
    elif args.action == 'predict':
        run_predict(args)
    elif args.action == 'plot':
        run_plot(args)
    elif args.action == 'visualize-labels':
        run_visualize_labels(args)
    elif args.action == 'download-dataset':
        run_download_dataset(args)
    else:
        print(f"Hành động không hợp lệ: {args.action}")
        sys.exit(1)

if __name__ == "__main__":
    main() 