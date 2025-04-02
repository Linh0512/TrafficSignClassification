# Ứng dụng nhận diện biển báo giao thông

Ứng dụng web nhận diện biển báo giao thông Việt Nam sử dụng YOLOv8 và FastAPI. Được thiết kế để deploy lên Vercel.

## Tính năng

- Nhận diện biển báo từ ảnh tải lên
- Nhận diện biển báo theo thời gian thực từ webcam
- Hiển thị thông tin chi tiết về các biển báo được phát hiện
- Cho phép điều chỉnh ngưỡng tin cậy (confidence threshold)

## Cấu trúc dự án

```
web/
├── app.py                 # Ứng dụng FastAPI
├── download_model.py      # Script tải mô hình
├── requirements.txt       # Danh sách các dependency
├── vercel.json            # Cấu hình deploy lên Vercel
├── static/                # Thư mục chứa CSS và JavaScript
└── templates/             # Thư mục chứa template HTML
```

## Yêu cầu

- Python 3.9+
- FastAPI
- Ultralytics YOLOv8
- OpenCV

## Cài đặt cục bộ

1. Clone repository
```bash
git clone https://github.com/username/traffic-sign-detection.git
cd traffic-sign-detection/web
```

2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

3. Tải model (nếu chưa có)
```bash
python download_model.py
```

4. Chạy ứng dụng
```bash
uvicorn app:app --reload
```

5. Truy cập ứng dụng tại http://localhost:8000

## Deploy lên Vercel

1. Đăng ký tài khoản Vercel (nếu chưa có) tại https://vercel.com/signup

2. Cài đặt Vercel CLI:
```bash
npm install -g vercel
```

3. Đăng nhập vào Vercel:
```bash
vercel login
```

4. Trong thư mục web, chạy lệnh:
```bash
vercel
```

5. Cấu hình biến môi trường trên Vercel dashboard:
   - `MODEL_PATH`: Đường dẫn tới file model (mặc định: "best.pt")
   - `MODEL_URL`: URL để tải model nếu không tìm thấy

## Lưu ý

- Khi deploy lên Vercel, model sẽ được tải tự động khi ứng dụng khởi động
- Chức năng webcam có thể không hoạt động trên một số trình duyệt hoặc thiết bị do giới hạn về quyền truy cập
- Đối với chức năng nhận diện qua webcam, cần có kết nối WebSocket ổn định

## Thay đổi model nhận diện

Để thay đổi model YOLOv8 được sử dụng, bạn có thể:

1. Thay đổi biến môi trường `MODEL_PATH` và `MODEL_URL` trong Vercel dashboard, hoặc
2. Thay thế trực tiếp file `best.pt` trong thư mục project

## Tác giả

- Your Name 