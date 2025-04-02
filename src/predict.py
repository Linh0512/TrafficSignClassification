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
    parser = argparse.ArgumentParser(description='Dự đoán biển báo giao thông bằng mô hình YOLOv8')
    parser.add_argument('--model', type=str, required=True, help='Đường dẫn đến model đã huấn luyện (*.pt)')
    parser.add_argument('--source', type=str, required=True, help='Đường dẫn đến ảnh hoặc thư mục chứa ảnh cần dự đoán')
    parser.add_argument('--data', type=str, default=os.path.join(root_dir, 'data/dataset/data.yaml'), help='File dữ liệu yaml')
    parser.add_argument('--conf', type=float, default=0.25, help='Ngưỡng tin cậy')
    parser.add_argument('--iou', type=float, default=0.7, help='Ngưỡng IoU cho NMS')
    parser.add_argument('--device', type=str, default='', help='cuda device (0, 1, ...) hoặc cpu')
    parser.add_argument('--save-txt', action='store_true', help='Lưu kết quả nhận diện dưới dạng văn bản')
    parser.add_argument('--save-conf', action='store_true', help='Lưu ngưỡng tin cậy cùng với dự đoán')
    parser.add_argument('--view-img', action='store_true', help='Hiển thị ảnh kết quả')
    parser.add_argument('--output', type=str, default=os.path.join(root_dir, 'results'), help='Thư mục lưu kết quả')
    
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

def load_class_names(data_yaml):
    """Tải tên các lớp từ file YAML"""
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    class_names = data['names']
    return class_names

def predict_and_draw(model, image_path, class_names, conf_thres, iou_thres, device, output_dir, view_img, save_txt, save_conf):
    """Dự đoán và vẽ bounding box lên ảnh"""
    # Tải ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh từ {image_path}")
        return
    
    # Dự đoán với model
    results = model(image_path, conf=conf_thres, iou=iou_thres, device=device)[0]
    
    # Lấy thông tin dự đoán
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    
    # Tạo kết quả ảnh
    image_result = image.copy()
    
    # Vẽ bounding box lên ảnh
    for box, conf, class_id in zip(boxes, confs, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        class_name = class_names[class_id]
        
        # Vẽ bounding box
        cv2.rectangle(image_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Thêm label với ngưỡng tin cậy
        label = f"{class_name} {conf:.2f}"
        cv2.putText(image_result, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Lưu kết quả ảnh
    file_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, file_name)
    cv2.imwrite(output_path, image_result)
    
    # Hiển thị ảnh nếu được yêu cầu
    if view_img:
        cv2.imshow('Result', image_result)
        cv2.waitKey(0)
    
    # Lưu kết quả dưới dạng văn bản nếu được yêu cầu
    if save_txt:
        txt_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.txt")
        with open(txt_path, 'w') as f:
            for box, conf, class_id in zip(boxes, confs, class_ids):
                x1, y1, x2, y2 = box
                # Format: <class_id> <x1> <y1> <x2> <y2> [<conf>]
                line = f"{class_id} {x1} {y1} {x2} {y2}"
                if save_conf:
                    line += f" {conf}"
                f.write(line + '\n')
    
    return image_result, len(boxes)

def main():
    """Hàm chính"""
    args = parse_args()
    
    # Tạo thư mục lưu kết quả
    os.makedirs(args.output, exist_ok=True)
    
    # Tải model
    model = YOLO(args.model)
    
    # Tải thông tin các class
    class_names = load_class_names(args.data)
    
    # Kiểm tra device
    device = args.device
    if device == '':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print(f"Dự đoán trên device: {device}")
    
    # Kiểm tra nguồn đầu vào
    source = args.source
    if os.path.isfile(source):
        # Dự đoán trên một ảnh
        image_paths = [source]
    elif os.path.isdir(source):
        # Dự đoán trên thư mục ảnh
        image_paths = [os.path.join(source, f) for f in os.listdir(source) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    else:
        print(f"Nguồn đầu vào không hợp lệ: {source}")
        return
    
    if not image_paths:
        print(f"Không tìm thấy ảnh trong {source}")
        return
    
    print(f"Tìm thấy {len(image_paths)} ảnh để dự đoán")
    
    # Dự đoán trên từng ảnh
    total_objects = 0
    for image_path in image_paths:
        print(f"Dự đoán trên ảnh: {image_path}")
        _, num_objects = predict_and_draw(
            model=model,
            image_path=image_path,
            class_names=class_names,
            conf_thres=args.conf,
            iou_thres=args.iou,
            device=device,
            output_dir=args.output,
            view_img=args.view_img,
            save_txt=args.save_txt,
            save_conf=args.save_conf
        )
        total_objects += num_objects
    
    print(f"\nĐã hoàn thành dự đoán trên {len(image_paths)} ảnh")
    print(f"Tổng số đối tượng đã phát hiện: {total_objects}")
    print(f"Kết quả đã được lưu vào thư mục: {args.output}")

def predict_on_video(args):
    """Dự đoán trên video"""
    # Tải model
    model = YOLO(args.model)
    
    # Tải thông tin các class
    class_names = load_class_names(args.data)
    
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