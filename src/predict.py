import os
import argparse
import yaml
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import sys

# Thêm thư mục gốc vào sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, root_dir)

def parse_args():
    """Xử lý các đối số dòng lệnh"""
    parser = argparse.ArgumentParser(description='Dự đoán biển báo giao thông bằng mô hình YOLOv12')
    parser.add_argument('--model', type=str, required=True, help='Đường dẫn đến model đã huấn luyện (*.pt)')
    parser.add_argument('--source', type=str, required=True, help='Đường dẫn đến ảnh hoặc thư mục chứa ảnh cần dự đoán')
    parser.add_argument('--data', type=str, default=os.path.join(root_dir, 'data/data.yaml'), help='File dữ liệu yaml')
    parser.add_argument('--conf', type=float, default=0.25, help='Ngưỡng tin cậy')
    parser.add_argument('--iou', type=float, default=0.7, help='Ngưỡng IoU cho NMS')
    parser.add_argument('--device', type=str, default='', help='cuda device (0, 1, ...) hoặc cpu')
    parser.add_argument('--save-txt', action='store_true', help='Lưu kết quả nhận diện dưới dạng văn bản')
    parser.add_argument('--save-conf', action='store_true', help='Lưu ngưỡng tin cậy cùng với dự đoán')
    parser.add_argument('--view-img', action='store_true', help='Hiển thị ảnh kết quả')
    parser.add_argument('--output', type=str, default=os.path.join(root_dir, 'outputs/results'), help='Thư mục lưu kết quả')
    
    args = parser.parse_args()
    
    # Kiểm tra đường dẫn đến mô hình
    if not os.path.exists(args.model):
        print(f"Lỗi: Không tìm thấy model tại '{args.model}'")
        print(f"Ví dụ đường dẫn model: {os.path.join(root_dir, 'runs/train/traffic_sign_detection/weights/best.pt')}")
        exit()
        
    # Kiểm tra đường dẫn đến nguồn ảnh/thư mục
    if not os.path.exists(args.source):
        print(f"Lỗi: Không tìm thấy nguồn dữ liệu tại '{args.source}'")
        print("Ví dụ:")
        print(f"  - Dự đoán một ảnh: --source {os.path.join(root_dir, 'data/dataset/test/images/Tên_ảnh.jpg')}")
        print(f"  - Dự đoán thư mục ảnh: --source {os.path.join(root_dir, 'data/dataset/test/images')}")
        print("  - Dự đoán video: --source đường/dẫn/đến/video.mp4")
        exit()
        
    return args

def load_classes(data_yaml):
    """Tải danh sách lớp từ file yaml"""
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def predict(args):
    """Thực hiện dự đoán bằng YOLOv12"""
    # Kiểm tra đầu vào
    if not os.path.exists(args.model):
        print(f"Lỗi: Không tìm thấy model tại {args.model}")
        return

    if not os.path.exists(args.source):
        print(f"Lỗi: Không tìm thấy nguồn ảnh tại {args.source}")
        return

    # Tải danh sách lớp
    classes = load_classes(args.data)
    print(f"Đã tải {len(classes)} lớp từ {args.data}")
    
    # Kiểm tra và tạo thư mục đầu ra
    os.makedirs(args.output, exist_ok=True)
    
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
    
    # Thực hiện dự đoán
    try:
        print(f"Dự đoán từ nguồn: {args.source}")
        results = model.predict(
            source=args.source,
            conf=args.conf,
            iou=args.iou,
            device=device,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            show=args.view_img,
            project=args.output,
            name='predictions',
            exist_ok=True,
            save=True  # Lưu kết quả
        )
        
        print(f"Kết quả dự đoán đã được lưu vào {os.path.join(args.output, 'predictions')}")
        
        # Phân tích kết quả (tuỳ chọn)
        analyze_results(results, classes)
        
    except Exception as e:
        print(f"Lỗi khi thực hiện dự đoán: {e}")
        
def analyze_results(results, classes):
    """Phân tích kết quả dự đoán"""
    class_count = {}
    total_objects = 0
    
    for r in results:
        # Lấy thông tin về các đối tượng được phát hiện
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls.item())
            cls_name = classes[cls_id]
            conf = box.conf.item()
            
            # Cập nhật số lượng
            if cls_name in class_count:
                class_count[cls_name] += 1
            else:
                class_count[cls_name] = 1
            
            total_objects += 1
    
    if total_objects > 0:
        print(f"\nĐã phát hiện tổng cộng {total_objects} đối tượng:")
        for cls_name, count in class_count.items():
            print(f"  - {cls_name}: {count} ({count/total_objects*100:.1f}%)")
    else:
        print("Không phát hiện đối tượng nào.")

def main():
    """Hàm chính"""
    args = parse_args()
    predict(args)

def predict_on_video(args):
    """Dự đoán trên video"""
    # Tải model
    model = YOLO(args.model)
    
    # Tải thông tin các class
    class_names = load_classes(args.data)
    
    # Kiểm tra device
    device = args.device
    if device == '':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print(f"Dự đoán trên device: {device}")
    
    # Mở video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Không thể mở video: {args.source}")
        return
    
    # Lấy thông tin video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Tạo video writer để lưu kết quả
    output_path = os.path.join(args.output, f"{Path(args.source).stem}_pred.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_objects = 0
    
    # Xử lý từng frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Dự đoán trên frame hiện tại
        results = model(frame, conf=args.conf, iou=args.iou, device=device)[0]
        
        # Lấy thông tin dự đoán
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        # Vẽ bounding box lên frame
        for box, conf, class_id in zip(boxes, confs, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            class_name = class_names[class_id]
            
            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Thêm label
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Đếm số đối tượng
        total_objects += len(boxes)
        
        # Hiển thị frame
        if args.view_img:
            cv2.imshow('Result', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Lưu frame vào video
        writer.write(frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Đã xử lý {frame_count} frames")
    
    # Giải phóng tài nguyên
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    print(f"\nĐã hoàn thành dự đoán trên video: {args.source}")
    print(f"Tổng số frames đã xử lý: {frame_count}")
    print(f"Tổng số đối tượng đã phát hiện: {total_objects}")
    print(f"Kết quả đã được lưu vào: {output_path}")

if __name__ == "__main__":
    args = parse_args()
    
    # Tạo thư mục lưu kết quả
    os.makedirs(args.output, exist_ok=True)
    
    # Kiểm tra xem đầu vào là ảnh hay video
    source = args.source
    if os.path.isfile(source) and source.lower().endswith(('.mp4', '.avi', '.mov')):
        predict_on_video(args)
    else:
        main() 