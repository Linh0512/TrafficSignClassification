import os
import sys
import yaml
import argparse
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Thêm thư mục gốc vào sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.append(root_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="Hiển thị và sửa đổi nhãn cho biển báo giao thông")
    parser.add_argument('--data', type=str, default='data/data.yaml',
                        help='Đường dẫn đến file data.yaml')
    parser.add_argument('--output_data', type=str, default='data/data_renamed.yaml',
                        help='Đường dẫn đến file data.yaml đầu ra với tên nhãn mới')
    parser.add_argument('--sample-per-class', type=int, default=5,
                        help='Số lượng ảnh mẫu để hiển thị cho mỗi lớp')
    return parser.parse_args()

def load_data_yaml(yaml_path):
    """Đọc file data.yaml"""
    yaml_path = os.path.join(root_dir, yaml_path)
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def save_data_yaml(data, yaml_path):
    """Lưu file data.yaml"""
    yaml_path = os.path.join(root_dir, yaml_path)
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"Đã lưu file cấu hình với tên nhãn mới vào {yaml_path}")

def get_label_images(data_yaml, yaml_path, split='train'):
    """Lấy danh sách ảnh và nhãn tương ứng cho mỗi lớp"""
    # Đường dẫn đến thư mục chứa file yaml
    yaml_dir = os.path.dirname(os.path.join(root_dir, yaml_path))
    
    # Xác định đường dẫn tương đối từ file yaml
    split_key = 'train' if split == 'train' else ('val' if split == 'val' else 'test')
    
    # Đường dẫn tương đối từ file yaml
    rel_images_path = data_yaml[split_key]
    
    # Đường dẫn tuyệt đối đến thư mục chứa ảnh
    if rel_images_path.startswith('./'):
        rel_images_path = rel_images_path[2:]  # Loại bỏ './' ở đầu
    
    # Tạo đường dẫn tuyệt đối tới thư mục ảnh và nhãn
    abs_images_dir = os.path.join(yaml_dir, rel_images_path)
    abs_labels_dir = abs_images_dir.replace('images', 'labels')
    
    print(f"Đang tìm ảnh trong thư mục: {abs_images_dir}")
    print(f"Đang tìm nhãn trong thư mục: {abs_labels_dir}")
    
    # Tạo từ điển để lưu ảnh cho mỗi lớp
    label_images = {i: [] for i in range(data_yaml['nc'])}
    
    # Tìm tất cả các file nhãn
    label_files = glob.glob(os.path.join(abs_labels_dir, '*.txt'))
    print(f"Tìm thấy {len(label_files)} file nhãn")
    
    for label_file in label_files:
        # Đọc file nhãn
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        # Lấy tên file ảnh tương ứng
        img_file = label_file.replace('labels', 'images').replace('.txt', '.jpg')
        if not os.path.exists(img_file):
            img_file = img_file.replace('.jpg', '.jpeg')
        if not os.path.exists(img_file):
            img_file = img_file.replace('.jpeg', '.png')
        if not os.path.exists(img_file):
            continue
        
        # Xử lý các nhãn trong file
        for line in lines:
            try:
                class_id = int(line.strip().split()[0])
                if 0 <= class_id < data_yaml['nc']:
                    label_images[class_id].append(img_file)
            except:
                continue
    
    # In số lượng ảnh cho mỗi nhãn
    for class_id in range(data_yaml['nc']):
        img_count = len(label_images[class_id])
        if img_count > 0:
            print(f"Lớp {class_id}: {data_yaml['names'][class_id]} - {img_count} ảnh")
    
    return label_images

def display_class_samples(data_yaml, label_images, samples_per_class=5):
    """Hiển thị các mẫu ảnh cho mỗi lớp"""
    names = data_yaml['names']
    
    for class_id in range(data_yaml['nc']):
        images = label_images.get(class_id, [])
        num_samples = min(samples_per_class, len(images))
        
        if num_samples == 0:
            print(f"Không có ảnh nào cho lớp {class_id}: {names[class_id]}")
            continue
        
        print(f"\nLớp {class_id}: {names[class_id]} - {len(images)} ảnh")
        
        # Hiển thị mẫu ảnh
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        
        if num_samples == 1:
            axes = [axes]
            
        for i in range(num_samples):
            img_path = images[i]
            print(f"Đọc ảnh: {img_path}")
            
            # Kiểm tra tệp tồn tại
            if not os.path.exists(img_path):
                print(f"File không tồn tại: {img_path}")
                continue
                
            # Đọc ảnh với xử lý lỗi
            try:
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Không thể đọc ảnh: {img_path}")
                    # Tạo ảnh màu đỏ để hiển thị lỗi
                    img = np.zeros((300, 300, 3), dtype=np.uint8)
                    img[:, :, 2] = 255  # đỏ
                    cv2.putText(img, "Image Error", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img)
                axes[i].set_title(f"ID: {class_id}")
                axes[i].axis('off')
            except Exception as e:
                print(f"Lỗi khi hiển thị ảnh {img_path}: {str(e)}")
                # Hiển thị lỗi trong ảnh
                img = np.zeros((300, 300, 3), dtype=np.uint8)
                img[:, :, 0] = 255  # đỏ
                axes[i].imshow(img)
                axes[i].set_title(f"Error: {class_id}")
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Hỏi người dùng về tên mới cho lớp
        new_name = input(f"Nhập tên mới cho lớp {names[class_id]} (nhấn Enter để giữ nguyên): ")
        if new_name:
            names[class_id] = new_name

def main():
    args = parse_args()
    
    # Đọc file data.yaml
    data_yaml = load_data_yaml(args.data)
    print(f"Đã tải file cấu hình với {data_yaml['nc']} lớp")
    
    # Lấy ảnh cho mỗi lớp
    print("Đang lấy ảnh cho mỗi lớp...")
    label_images = get_label_images(data_yaml, args.data, split='train')
    
    # Hiển thị mẫu ảnh và cho phép đổi tên nhãn
    print("\nHiển thị mẫu ảnh cho mỗi lớp và cho phép đổi tên nhãn")
    print("Nhấn 'q' để thoát chương trình")
    
    display_class_samples(data_yaml, label_images, args.sample_per_class)
    
    # Lưu file data.yaml mới
    save_data_yaml(data_yaml, args.output_data)
    print(f"Bạn có thể cập nhật file {args.data} bằng nội dung từ {args.output_data}")

if __name__ == "__main__":
    main() 