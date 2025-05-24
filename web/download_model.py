import os
import sys
import requests
import shutil
from tqdm import tqdm

# Thêm thư mục gốc vào sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# URL của model (thay thế bằng URL thực tế của model YOLOv12)
MODEL_URL = os.environ.get('MODEL_URL', 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt')

def download_file(url, destination):
    """Tải file từ URL và hiển thị tiến trình"""
    print(f"Đang tải model từ {url}...")
    
    # Tạo request với stream=True để tải từng phần
    response = requests.get(url, stream=True)
    
    # Lấy kích thước file
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    # Tạo thanh tiến trình
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    
    # Tải và ghi file
    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()
    
    # Kiểm tra kích thước file
    if total_size != 0 and progress_bar.n != total_size:
        print("Lỗi: Kích thước file không khớp!")
        return False
    
    print(f"File đã được tải về: {destination}")
    return True

def copy_trained_model(source, destination):
    """Copy mô hình đã train từ thư mục runs vào thư mục models"""
    try:
        print(f"Đang sao chép model đã train từ {source} đến {destination}...")
        shutil.copy2(source, destination)
        print(f"Đã sao chép model thành công!")
        return True
    except Exception as e:
        print(f"Lỗi khi sao chép model: {e}")
        return False

def main():
    # Đường dẫn đến mô hình đã train
    trained_model_path = os.path.join(root_dir, "runs", "train", "traffic_sign_detection_yolo12", "weights", "best.pt")
    
    # Đường dẫn lưu model trong thư mục models
    model_path = os.path.join(root_dir, "models", "best.pt")
    
    # Kiểm tra model đã tồn tại chưa
    if os.path.exists(model_path):
        print(f"Model đã tồn tại: {model_path}")
        return
    
    # Kiểm tra nếu mô hình đã train tồn tại, sao chép sang thư mục models
    if os.path.exists(trained_model_path):
        success = copy_trained_model(trained_model_path, model_path)
        if success:
            return
    
    # Nếu không tìm thấy mô hình đã train, tiến hành tải model từ web
    # Kiểm tra biến môi trường cấu hình
    model_url = os.environ.get('MODEL_URL')
    if not model_url:
        # Dùng model mặc định YOLOv12n
        model_url = MODEL_URL
        print(f"Không tìm thấy MODEL_URL, dùng model mặc định: {model_url}")
    
    # Tải model
    success = download_file(model_url, model_path)
    
    if success:
        print(f"Model đã được tải thành công: {model_path}")
    else:
        print(f"Không thể tải model từ {model_url}")
        sys.exit(1)

if __name__ == "__main__":
    main() 