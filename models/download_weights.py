import os
import sys
import requests
import argparse
from tqdm import tqdm
from pathlib import Path
import gdown

# Thêm thư mục gốc vào sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.append(root_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="Tải file weights cho mô hình Traffic Sign Detection")
    parser.add_argument('--model-name', type=str, default='best.pt',
                      help='Tên file weights (mặc định: best.pt)')
    return parser.parse_args()

def download_file(url, destination):
    """Tải file từ URL và hiển thị tiến trình"""
    print(f"Đang tải {os.path.basename(destination)}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as file, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)
    
    print(f"Đã tải thành công: {destination}")

def download_from_drive(model_name):
    """Tải weights từ Google Drive"""
    folder_id = "1_oGt6SJ9yOTzudmH_Elu1gEQitINYeS3"
    
    # Đường dẫn lưu file
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    destination = os.path.join(models_dir, model_name)
    
    try:
        print(f"Đang tải {model_name} từ Google Drive...")
        # Tải toàn bộ folder từ Google Drive
        gdown.download_folder(id=folder_id, output=models_dir, quiet=False)
        
        # Kiểm tra xem file cần có tồn tại không
        if not os.path.exists(destination):
            # Tìm và đổi tên file .pt đầu tiên nếu cần
            for file in os.listdir(models_dir):
                if file.endswith('.pt') and file != model_name:
                    os.rename(os.path.join(models_dir, file), destination)
                    break
        
        if os.path.exists(destination):
            print(f"Đã tải thành công: {destination}")
            return True
        else:
            print(f"Không tìm thấy file {model_name} trong thư mục đã tải")
            return False
    except Exception as e:
        print(f"Lỗi khi tải từ Google Drive: {str(e)}")
        return False

def download_pretrained_yolov8():
    """Tải mô hình YOLOv8 pretrained từ Ultralytics"""
    try:
        from ultralytics import YOLO
        
        print("Đang tải mô hình YOLOv8n pretrained...")
        model = YOLO('yolov8n.pt')
        
        # Lưu mô hình vào thư mục models
        destination = os.path.join(root_dir, "models", "yolov8n.pt")
        import shutil
        shutil.copy(model.ckpt_path, destination)
        
        print(f"Đã tải thành công mô hình pretrained YOLOv8n: {destination}")
        return True
    except Exception as e:
        print(f"Lỗi khi tải mô hình pretrained YOLOv8n: {str(e)}")
        return False

def main():
    args = parse_args()
    
    # Tạo thư mục models nếu chưa tồn tại
    os.makedirs(os.path.join(root_dir, "models"), exist_ok=True)
    
    # Tải mô hình YOLOv8 pretrained
    download_pretrained_yolov8()
    
    # Tải weights từ Google Drive
    success = download_from_drive(args.model_name)
    
    if not success:
        print("\nHướng dẫn cài đặt manual:")
        print("1. Tải file weights từ link: https://drive.google.com/drive/folders/1_oGt6SJ9yOTzudmH_Elu1gEQitINYeS3?usp=sharing")
        print("2. Đặt file weights vào thư mục models/")
        print("3. Đảm bảo rằng file weights có tên là 'best.pt' hoặc tên được chỉ định trong lệnh chạy")
        
    print("\nSử dụng weights để predict:")
    print(f"python main.py --action predict --model models/{args.model_name} --source <đường_dẫn_ảnh>")

if __name__ == "__main__":
    main()