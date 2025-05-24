# YOLOv12 Object Detection Project

Dự án nhận diện đối tượng sử dụng YOLOv12 với dataset từ Roboflow.

## 📁 Cấu trúc dự án (Đã được tái cấu trúc)

```
FinalProject_Yolov12/
├── src/                          # Mã nguồn chính
│   ├── main.py                   # File chính để chạy tất cả chức năng
│   ├── train.py                  # Huấn luyện model
│   ├── evaluate.py               # Đánh giá model
│   ├── predict.py                # Dự đoán với model đã huấn luyện
│   ├── visualize_labels.py       # Hiển thị và chỉnh sửa nhãn
│   ├── download_dataset.py       # Tải dataset từ Roboflow
│   ├── eda.py                    # Phân tích khám phá dữ liệu
│   ├── plot_results.py           # Vẽ biểu đồ kết quả
│   └── update_settings.py        # Cập nhật cài đặt
├── models/                       # Models YOLOv12
│   ├── best_yolo12.pt           # Model tốt nhất đã huấn luyện
│   ├── yolo12n.pt               # Model base YOLOv12n
│   ├── best.pt                  # Model backup
│   └── download_weights.py      # Script tải weights
├── data/                        # Dataset chính
│   ├── train/                   # Ảnh và nhãn huấn luyện
│   │   ├── images/
│   │   └── labels/
│   ├── valid/                   # Ảnh và nhãn validation
│   │   ├── images/
│   │   └── labels/
│   ├── test/                    # Ảnh và nhãn test
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml               # Cấu hình dataset
├── outputs/                     # Kết quả và outputs (đã gộp)
│   ├── runs/                    # Kết quả training/validation
│   ├── results/                 # Kết quả prediction
│   ├── visualizations/          # Biểu đồ và visualization
│   └── eda_results/            # Kết quả phân tích EDA
├── web/                         # Web application (riêng biệt)
│   ├── app.py                   # Flask/FastAPI app
│   ├── templates/               # HTML templates
│   ├── static/                  # CSS, JS, images
│   ├── requirements.txt         # Dependencies cho web
│   └── README.md               # Hướng dẫn web app
├── requirements.txt             # Dependencies chính
├── .gitignore                  # Git ignore (đã cập nhật)
└── README.md                   # File này
```

## 🚀 Khởi động project

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Tải dataset (nếu chưa có)
```bash
python src/main.py --mode download --data data/data.yaml
```

### 3. Phân tích dữ liệu (EDA)
```bash
python src/main.py --mode eda --data data/data.yaml
```

### 4. Huấn luyện model (50 epochs)
```bash
python src/main.py --mode train --model models/yolo12n.pt --data data/data.yaml --epochs 50
```

### 5. Đánh giá model
```bash
python src/main.py --mode evaluate --model models/best_yolo12.pt --data data/data.yaml
```

### 6. Dự đoán
```bash
python src/main.py --mode predict --model models/best_yolo12.pt --source path/to/image --data data/data.yaml
```

## 🔧 Các tính năng chính

- **Training**: Huấn luyện model YOLOv12 với custom dataset
- **Evaluation**: Đánh giá hiệu suất model với metrics chi tiết
- **Prediction**: Dự đoán trên ảnh mới với visualization
- **EDA**: Phân tích khám phá dữ liệu với các biểu đồ
- **Visualization**: Hiển thị và chỉnh sửa nhãn dataset
- **Web Interface**: Giao diện web để sử dụng model (thư mục `web/`)

## 📊 Dataset

Dataset chứa 59 classes của các đối tượng thực phẩm và đồ gia dụng:
- **Training**: 2280+ images
- **Validation**: ~400 images  
- **Test**: ~200 images

## 🎯 Models

Dự án sử dụng YOLOv12 với các phiên bản:
- `yolo12n.pt`: Model base YOLOv12 nano
- `best_yolo12.pt`: Model tốt nhất
- `best.pt`: Model backup

## 📈 Kết quả

Tất cả kết quả được lưu trong thư mục `outputs/`:
- Training logs và metrics: `outputs/runs/`
- Prediction results: `outputs/results/`
- Visualization plots: `outputs/visualizations/`
- EDA analysis: `outputs/eda_results/`

## 🌐 Web Application

Để chạy web interface:
```bash
cd web
pip install -r requirements.txt
python app.py
```

## 📄 License

MIT License