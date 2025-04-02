import os
import sys
import json
import uuid
import logging
from typing import List, Optional
from pathlib import Path
import tempfile
import time
import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Thêm thư mục gốc vào sys.path để có thể import các module khác
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# Thử import mô hình YOLOv8
try:
    from ultralytics import YOLO
except ImportError:
    os.system('pip install ultralytics')
    from ultralytics import YOLO

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="Traffic Sign Detection API")

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thư mục chứa các file tĩnh
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# Mount thư mục static
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Thiết lập templates
templates = Jinja2Templates(directory=templates_dir)

# Thông tin về nhãn
class_descriptions = {
    # Thêm thông tin mô tả cho mỗi nhãn ở đây
    # Ví dụ: 'DP.135': 'Biển báo cấm đỗ xe',
    'DP.135': 'biển báo hết mọi lệnh cấm',
    'P.102': 'Cấm đi ngược chiều',
    'P.103a': 'Cấm xe ô tô',
    'P.103b': 'Cấm xe ôtô rẽ phải',
    'P.103c': 'Cấm xe ôtô rẽ trái',
    'P.104': 'Cấm xe máy',
    'P.106a': 'Cấm xe tải',
    'P.106b': 'Cấm xe tải trên 2,5T',
    'P.107a': 'Cấm ôtô khách',
    'P.112': 'Cấm người đi bộ',
    'P.115': 'Hạn chế tải trọng toàn bộ xe',
    'P.117': 'Hạn chế chiều cao',
    'P.123a': 'Cấm rẽ trái',
    'P.123b': 'Cấm rẽ phải',
    'P.124a': 'Cấm quay xe',
    'P.124b': 'Cấm ôtô quay đầu xe',
    'P.124c': 'Cấm rẽ trái và quay đầu xe',
    'P.125': 'Cấm vượt',
    'P.127': 'Tốc độ tối đa cho phép',
    'P.128': 'Cấm bóp còi',
    'P.130': 'Cấm dừng xe và đỗ xe',
    'P.131a': 'Cấm đỗ xe',
    'P.137': 'Cấm rẽ trái và rẽ phải',
    'P.245a': 'Đi chậm',
    'R.301c': 'Phương tiện chỉ được rẽ trái',
    'R.301d': 'Phương tiện chỉ được rẽ phải',
    'R.301e': 'Phương tiện chỉ được rẽ trái',
    'R.302a': 'Phương tiện đi vòng sang phải',
    'R.302b': 'Phương tiện đi vòng sang trái',
    'R.303': 'Nơi giao nhau chạy theo vòng xuyến',
    'R.407a': 'Đường 1 chiều đi thẳng',
    'R.409': 'Biển chỉ dẫn điểm quay xe R',
    'R.425': 'Biển báo bệnh viện',
    'R.434': 'Biển báo Bến xe buýt',
    'S.509a': 'Biển báo chiều cao an toàn',
    'W.201a': 'Biển báo chỗ ngoặc nguy hiểm vòng bên trái',
    'W.201b': 'Biển báo chỗ ngoặc nguy hiểm vòng bên phải',
    'W.202a': 'Biển báo nhiều chỗ ngoặc nguy hiểm liên tiếp, chỗ ngoặt đầu tiên hướng vòng bên trái',
    'W.202b': 'Biển báo nhiều chỗ ngoặc nguy hiểm liên tiếp, chỗ ngoặt đầu tiên hướng vòng bên phải',
    'W.203b': 'Biển báo đường bị thu hẹp bên trái',
    'W.203c': 'Biển báo đường bị thu hẹp bên phải',
    'W.205a': 'Đường giao nhau theo kiểu chữ thập',
    'W.205b': 'Đường giao nhau có nhánh bên phải',
    'W.205d': 'Đường giao nhau dạng chữ T',
    'W.207a': 'Biển báo giao nhau với đường không ưu tiên',
    'W.207b': 'Biển báo giao nhau với đường không ưu tiên ở bên phải',
    'W.207c': 'Biển báo giao nhau với đường không ưu tiên ở bên trái',
    'W.208': 'Biển báo giao nhau với đường ưu tiên',
    'W.209': 'Biển báo giao nhau có tính hiệu đèn',
    'W.210': 'Biển báo giao nhau với đường sắt có rào chắn',
    'W.219': 'Biển Báo Dốc Xuống Nguy Hiểm',
    'W.221b': 'Biể báo đường có gồ giảm tốc',
    'W.224': 'Biển báo đường có người đi bộ cắt ngang',
    'W.225': 'Biển báo đoạn đường thường có trẻ em đi qua',
    'W.227': 'Biển báo công trường đang thi công',
    'W.233': 'Biển báo nguy hiểm khác',
    'W.235': 'Biển báo đường đôi',
    'W.245a': 'Biển báo đi chậm',
}

# Khởi tạo biến để theo dõi các kết nối WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_text(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

# Khởi tạo quản lý kết nối
manager = ConnectionManager()

# Mô hình YOLOv8
class ModelManager:
    def __init__(self):
        self.model = None
        # Đường dẫn mặc định đến model đã train
        self.model_path = os.path.join(root_dir, "models", "best.pt")
        
    def load_model(self):
        if self.model is None:
            try:
                # Trước tiên, tìm model trong thư mục models
                if os.path.exists(self.model_path):
                    self.model = YOLO(self.model_path)
                    logger.info(f"Model loaded from {self.model_path}")
                # Nếu không tìm thấy, tìm trong thư mục hiện tại
                elif os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")):
                    self.model = YOLO(os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt"))
                    logger.info("Model loaded from current directory")
                # Nếu không tìm thấy, dùng model mặc định yolov8n.pt
                else:
                    self.model = YOLO("yolov8n.pt")
                    logger.info("Using default YOLOv8n model")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                # Dùng model mặc định nếu có lỗi
                self.model = YOLO("yolov8n.pt")
                logger.info("Using default YOLOv8n model due to error")
        return self.model

# Khởi tạo quản lý mô hình và tải model ngay khi khởi tạo
model_manager = ModelManager()
model_manager.load_model()

# Trang chủ
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API nhận diện ảnh tải lên
@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...), conf: float = Form(0.25)):
    # Tạo thư mục tạm để lưu ảnh tải lên
    with tempfile.TemporaryDirectory() as temp_dir:
        # Đọc file ảnh tải lên
        contents = await file.read()
        image_path = os.path.join(temp_dir, file.filename)
        
        # Lưu file ảnh
        with open(image_path, "wb") as f:
            f.write(contents)
        
        # Tải mô hình nếu chưa tải
        model = model_manager.load_model()
        
        # Thực hiện dự đoán
        results = model(image_path, conf=conf)
        
        # Xử lý kết quả
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Lấy thông tin
                cls = int(box.cls[0])
                class_name = r.names[cls]
                conf_val = float(box.conf[0])
                
                # Thêm vào danh sách
                detections.append({
                    "class": class_name,
                    "confidence": conf_val,
                    "description": class_descriptions.get(class_name, "")
                })
        
        # Vẽ kết quả
        result = results[0].plot()
        
        # Chuyển kết quả thành base64
        _, buffer = cv2.imencode('.jpg', result)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Trả về kết quả
        return {
            "image": image_base64,
            "detections": detections
        }

# WebSocket endpoint cho webcam
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    # Đảm bảo model đã được tải
    model = model_manager.load_model()
    
    # Biến để theo dõi lớp đã phát hiện
    detected_classes = set()
    
    try:
        while True:
            # Nhận dữ liệu từ client
            data = await websocket.receive_text()
            
            try:
                # Parse dữ liệu JSON
                data_json = json.loads(data)
                
                # Lấy ảnh và ngưỡng tin cậy
                image_base64 = data_json.get("image", "")
                conf = float(data_json.get("conf", 0.25))
                
                # Chuyển base64 thành numpy array
                image_bytes = base64.b64decode(image_base64)
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                # Kiểm tra nếu ảnh hợp lệ
                if image is None or image.size == 0:
                    continue
                
                # Thực hiện dự đoán
                results = model(image, conf=conf, verbose=False)
                
                # Lấy các lớp hiện tại
                current_classes = set()
                detections = []
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # Lấy thông tin
                        cls = int(box.cls[0])
                        class_name = r.names[cls]
                        conf_val = float(box.conf[0])
                        
                        # Thêm vào set lớp hiện tại
                        current_classes.add(class_name)
                        
                        # Thêm vào danh sách
                        detections.append({
                            "class": class_name,
                            "confidence": conf_val,
                            "description": class_descriptions.get(class_name, "")
                        })
                
                # Tìm các lớp mới
                new_classes = current_classes - detected_classes
                
                # Lọc ra chỉ các phát hiện thuộc lớp mới
                new_detections = [d for d in detections if d["class"] in new_classes]
                
                # Cập nhật các lớp đã phát hiện
                detected_classes.update(new_classes)
                
                # Nếu có phát hiện mới, gửi kết quả về
                if new_detections:
                    # Vẽ kết quả
                    result = results[0].plot()
                    
                    # Chuyển kết quả thành base64
                    _, buffer = cv2.imencode('.jpg', result)
                    result_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Gửi kết quả
                    await websocket.send_text(json.dumps({
                        "image": result_base64,
                        "detections": new_detections
                    }))
                
                # Nếu không còn phát hiện nào, reset danh sách đã phát hiện
                if not current_classes:
                    detected_classes.clear()
                    
            except Exception as e:
                logger.error(f"Error processing webcam frame: {e}")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# API để lấy thông tin mô hình
@app.get("/model-info")
async def get_model_info():
    model = model_manager.load_model()
    names = model.names
    return {
        "model": model_manager.model_path,
        "classes": names
    }

# Hàm khởi động
@app.on_event("startup")
async def startup_event():
    # Tải mô hình
    model_manager.load_model()

# Dùng cho việc kiểm tra trạng thái API
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Chạy ứng dụng (khi test cục bộ)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 