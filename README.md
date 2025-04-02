# Hệ thống nhận diện biển báo giao thông Việt Nam

Hệ thống phát hiện, phân loại biển báo giao thông Việt Nam sử dụng YOLOv8.

## Cấu trúc dự án

```
project/
├── data/                      # Thư mục dữ liệu
│   └── dataset/               # Cấu hình và thông tin về dataset
├── dataset/                   # Dữ liệu huấn luyện (không đẩy lên Git)
├── models/                    # Thư mục chứa các mô hình
│   ├── download_weights.py    # Script tải weights mô hình
│   └── .gitkeep               # Đảm bảo thư mục được theo dõi bởi Git
├── results/                   # Kết quả dự đoán
├── runs/                      # Thư mục lưu quá trình huấn luyện
├── src/                       # Mã nguồn
│   ├── eda.py                 # Phân tích dữ liệu
│   ├── train.py               # Huấn luyện mô hình
│   ├── evaluate.py            # Đánh giá mô hình
│   ├── predict.py             # Dự đoán với mô hình đã huấn luyện
│   ├── plot_results.py        # Vẽ đồ thị kết quả huấn luyện
│   ├── update_settings.py     # Cập nhật cài đặt Ultralytics
│   ├── visualize_labels.py    # Hiển thị và sửa tên nhãn
│   └── download_dataset.py    # Tải dữ liệu từ Roboflow
├── visualizations/            # Thư mục lưu các hình ảnh và đồ thị
├── web/                       # Ứng dụng web deploy lên Vercel
│   ├── app.py                 # FastAPI app
│   ├── requirements.txt       # Các thư viện cần thiết
│   └── static/                # CSS, JavaScript, và các file tĩnh
├── main.py                    # File điều khiển chính
├── requirements.txt           # Danh sách các thư viện cần thiết
└── README.md                  # Hướng dẫn sử dụng
```

## Cài đặt

1. Clone dự án:
```bash
git clone https://github.com/Linh0512/traffic-sign-detection.git
cd traffic-sign-detection
```

2. Cài đặt thư viện cần thiết:
```bash
pip install -r requirements.txt
```

3. Tải weights mô hình:
```bash
python models/download_weights.py --source drive
```

4. Cập nhật đường dẫn dataset cho Ultralytics:
```bash
python src/update_settings.py
```

## Sử dụng cách 1: Sử dụng các script riêng

### Phân tích dữ liệu (EDA)
```bash
python src/eda.py
```
Kết quả phân tích sẽ được lưu trong thư mục `visualizations/eda_results/`.

### Huấn luyện mô hình
```bash
python src/train.py --model models/yolov8n.pt --epochs 50 --batch-size 16
```

### Đánh giá mô hình
```bash
python src/evaluate.py --model runs/train/traffic_sign_detection/weights/best.pt --visualize
```

### Dự đoán với mô hình đã huấn luyện
```bash
python src/predict.py --model runs/train/traffic_sign_detection/weights/best.pt --source data/dataset/test/images
```

### Vẽ biểu đồ kết quả huấn luyện
```bash
python src/plot_results.py --result-path runs/train/traffic_sign_detection
```

### Hiển thị và thay đổi tên nhãn
```bash
python src/visualize_labels.py
```

## Sử dụng cách 2: Chạy thông qua file main.py

### Phân tích dữ liệu (EDA)
```bash
python main.py --action eda
```

### Huấn luyện mô hình
```bash
python main.py --action train --model models/yolov8n.pt --epochs 50 --batch-size 16
```

### Đánh giá mô hình
```bash
python main.py --action evaluate --model runs/train/traffic_sign_detection/weights/best.pt --visualize
```

### Dự đoán với mô hình đã huấn luyện
```bash
python main.py --action predict --model runs/train/traffic_sign_detection/weights/best.pt --source data/dataset/test/images
```

### Vẽ biểu đồ kết quả huấn luyện
```bash
python main.py --action plot --result-path runs/train/traffic_sign_detection
```

### Hiển thị và thay đổi tên nhãn
```bash
python main.py --action visualize-labels
```

## Deploy lên Vercel

Ứng dụng web được triển khai trên Vercel cho phép người dùng:
- Sử dụng webcam để nhận diện biển báo giao thông theo thời gian thực
- Tải lên ảnh để phân tích và nhận diện biển báo giao thông

Để truy cập ứng dụng web:
- URL: [https://traffic-sign-detection.vercel.app](https://traffic-sign-detection.vercel.app)

## Lưu ý khi clone dự án

Khi clone dự án từ GitHub, do các file dữ liệu và weights không được đẩy lên (do kích thước lớn), bạn cần:

1. Tải mô hình weights:
   ```bash
   python models/download_weights.py
   ```

2. Nếu muốn huấn luyện lại mô hình, bạn cần tải dữ liệu từ Roboflow:
   ```bash
   python src/download_dataset.py
   ```

## Công cụ và thư viện sử dụng

- YOLOv8: Thuật toán phát hiện đối tượng
- Ultralytics: Framework triển khai YOLOv8
- OpenCV: Xử lý ảnh
- Matplotlib: Vẽ đồ thị
- FastAPI: Framework web API 
- Vercel: Nền tảng triển khai ứng dụng

## Tác giả

- Trần Qui Linh
- Huỳnh Đăng Khoa