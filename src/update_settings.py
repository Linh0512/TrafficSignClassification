import os
import json
from pathlib import Path
import sys

# Thêm thư mục gốc vào sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, root_dir)

# Đặt lại đường dẫn datasets trong Ultralytics settings
settings_path = Path(os.path.expanduser("~")) / "AppData" / "Roaming" / "Ultralytics" / "settings.json"

if settings_path.exists():
    # Đọc cài đặt hiện tại
    with open(settings_path, 'r') as f:
        settings = json.load(f)
    
    # Đường dẫn dự án hiện tại
    current_dir = root_dir
    
    # Cập nhật đường dẫn datasets
    settings['datasets_dir'] = current_dir
    
    # Lưu cài đặt
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=4)
    
    print(f"Đã cập nhật cài đặt Ultralytics đường dẫn datasets: {current_dir}")
else:
    # Tạo settings mới
    os.makedirs(settings_path.parent, exist_ok=True)
    settings = {
        "datasets_dir": root_dir
    }
    
    # Lưu cài đặt
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=4)
    
    print(f"Đã tạo cài đặt Ultralytics mới với đường dẫn datasets: {root_dir}")

print("Hãy chạy lại file src/train.py") 