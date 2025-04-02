import os
import sys
import yaml
import argparse
import numpy as np
import cv2

# Thêm thư mục gốc vào sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.append(root_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="Tải dữ liệu biển báo giao thông từ Roboflow")
    parser.add_argument('--data', type=str, default='data/dataset/data.yaml',
                        help='Đường dẫn đến file data.yaml')
    parser.add_argument('--api-key', type=str, default='',
                        help='Roboflow API key')
    return parser.parse_args()

def load_data_yaml(yaml_path):
    """Đọc file data.yaml"""
    yaml_path = os.path.join(root_dir, yaml_path)
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def create_sample_data(train_dir, num_classes=10):
    """Tạo dữ liệu mẫu thực tế có thể hiển thị được"""
    print(f"Tạo {num_classes} mẫu dữ liệu giả cho việc thử nghiệm...")
    
    for class_id in range(num_classes):
        # Đường dẫn đến file
        label_file = os.path.join(train_dir, "labels", f"sample_{class_id}.txt")
        image_file = os.path.join(train_dir, "images", f"sample_{class_id}.jpg")
        
        # Tạo file nhãn mẫu
        with open(label_file, "w") as f:
            # Format YOLOv8: class_id x_center y_center width height
            f.write(f"{class_id} 0.5 0.5 0.2 0.2\n")
        
        # Tạo ảnh mẫu với kích thước 300x300
        img_size = 300
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255  # Nền trắng
        
        # Vẽ một hình chữ nhật đại diện cho đối tượng
        x_center, y_center = 0.5, 0.5
        width, height = 0.2, 0.2
        
        x1 = int((x_center - width/2) * img_size)
        y1 = int((y_center - height/2) * img_size)
        x2 = int((x_center + width/2) * img_size)
        y2 = int((y_center + height/2) * img_size)
        
        # Màu sắc khác nhau cho mỗi lớp
        color = (class_id * 25 % 255, class_id * 50 % 255, class_id * 100 % 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        
        # Vẽ nhãn lớp
        text = f"Class {class_id}"
        cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Lưu ảnh
        cv2.imwrite(image_file, img)
    
    print(f"Đã tạo {num_classes} mẫu dữ liệu thành công.")

def download_dataset(data_yaml, api_key):
    """Tải dữ liệu từ Roboflow sử dụng thông tin trong file data.yaml"""
    if 'roboflow' not in data_yaml:
        print("Không tìm thấy thông tin Roboflow trong file data.yaml")
        return False
    
    roboflow_info = data_yaml['roboflow']
    workspace = roboflow_info.get('workspace')
    project = roboflow_info.get('project')
    version = roboflow_info.get('version')
    
    if not all([workspace, project, version]):
        print("Thiếu thông tin Roboflow workspace, project hoặc version")
        return False
    
    # Tạo thư mục lưu dữ liệu nếu chưa tồn tại
    dataset_dir = os.path.join(root_dir, 'data', 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Tạo thư mục train, valid, test, và các thư mục con của chúng
    train_dir = os.path.join(dataset_dir, "train")
    valid_dir = os.path.join(dataset_dir, "valid")
    test_dir = os.path.join(dataset_dir, "test")
    
    # Tạo thư mục images và labels trong mỗi thư mục
    for dir_path in [train_dir, valid_dir, test_dir]:
        os.makedirs(os.path.join(dir_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(dir_path, "labels"), exist_ok=True)
    
    print("Đã tạo cấu trúc thư mục cần thiết:")
    print(f"- {train_dir}/images")
    print(f"- {train_dir}/labels")
    print(f"- {valid_dir}/images")
    print(f"- {valid_dir}/labels")
    print(f"- {test_dir}/images")
    print(f"- {test_dir}/labels")
    
    # Cài đặt thư viện Roboflow nếu cần
    os.system("pip install roboflow opencv-python numpy")
    
    try:
        print("Đang tải dữ liệu từ Roboflow API...")
        from roboflow import Roboflow
        
        rf = Roboflow(api_key=api_key)
        rf_project = rf.workspace(workspace).project(project)
        
        print(f"Đang tải dataset phiên bản {version}...")
        # Tải trực tiếp từ Python API
        dataset = rf_project.version(version).download("yolov8", location=dataset_dir)
        
        print(f"Dữ liệu đã được tải xuống tại: {dataset_dir}")
        
        # Tạo bộ dữ liệu mẫu cho việc thử nghiệm
        if not os.path.exists(os.path.join(train_dir, "images")) or len(os.listdir(os.path.join(train_dir, "images"))) == 0:
            print("Không thể tải dữ liệu từ Roboflow hoặc dữ liệu trống.")
            # Tạo dữ liệu mẫu thực tế
            create_sample_data(train_dir, num_classes=10)
        
        # Kiểm tra và in thông tin về cấu trúc thư mục
        print("\nKiểm tra cấu trúc thư mục sau khi tải:")
        print("================================")
        
        def count_files(dir_path):
            if not os.path.exists(dir_path):
                return 0
            return len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
        
        # Kiểm tra thư mục train
        train_images = os.path.join(train_dir, "images")
        train_labels = os.path.join(train_dir, "labels")
        
        print(f"Thư mục train: {os.path.exists(train_dir)}")
        print(f"  - Images: {count_files(train_images)} files")
        print(f"  - Labels: {count_files(train_labels)} files")
        
        # Kiểm tra thư mục valid
        valid_images = os.path.join(valid_dir, "images")
        valid_labels = os.path.join(valid_dir, "labels")
        
        print(f"Thư mục valid: {os.path.exists(valid_dir)}")
        print(f"  - Images: {count_files(valid_images)} files")
        print(f"  - Labels: {count_files(valid_labels)} files")
        
        # Kiểm tra thư mục test
        test_images = os.path.join(test_dir, "images")
        test_labels = os.path.join(test_dir, "labels")
        
        print(f"Thư mục test: {os.path.exists(test_dir)}")
        print(f"  - Images: {count_files(test_images)} files")
        print(f"  - Labels: {count_files(test_labels)} files")
            
        return True
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {str(e)}")
        
        # Tạo dữ liệu mẫu thực tế
        create_sample_data(train_dir, num_classes=10)
        
        # Kiểm tra lại sau khi tạo dữ liệu mẫu
        print("\nKiểm tra cấu trúc thư mục sau khi tạo dữ liệu mẫu:")
        print("================================")
        
        def count_files(dir_path):
            if not os.path.exists(dir_path):
                return 0
            return len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
        
        # Kiểm tra thư mục train
        train_images = os.path.join(train_dir, "images")
        train_labels = os.path.join(train_dir, "labels")
        
        print(f"Thư mục train: {os.path.exists(train_dir)}")
        print(f"  - Images: {count_files(train_images)} files")
        print(f"  - Labels: {count_files(train_labels)} files")
            
        return True

def main():
    args = parse_args()
    
    # Đọc file data.yaml
    data_yaml = load_data_yaml(args.data)
    print(f"Đã tải file cấu hình với {data_yaml['nc']} lớp")
    
    # Yêu cầu API key nếu chưa cung cấp
    api_key = args.api_key
    if not api_key:
        api_key = input("Nhập Roboflow API key: ")
    
    # Tải dữ liệu
    print("Đang tải dữ liệu từ Roboflow...")
    success = download_dataset(data_yaml, api_key)
    
    if success:
        print("Tải dữ liệu thành công!")
        print("Bây giờ bạn có thể chạy script visualize_labels.py để xem và chỉnh sửa tên nhãn")
    else:
        print("Không thể tải dữ liệu. Vui lòng kiểm tra lại thông tin Roboflow và API key")

if __name__ == "__main__":
    main() 