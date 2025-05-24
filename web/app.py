import os
import sys
import json
import uuid
import logging
import shutil
from typing import List, Optional
from pathlib import Path
import tempfile
import time
import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io

# Thêm thư mục gốc vào sys.path để có thể import các module khác
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# Thử import mô hình YOLO
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
temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")

# Đảm bảo thư mục temp tồn tại
os.makedirs(temp_dir, exist_ok=True)

# Mount thư mục static
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Thiết lập templates
templates = Jinja2Templates(directory=templates_dir)

# Lớp model cho frame video
class VideoFrame(BaseModel):
    image: str
    conf: float = 0.25
    stabilize: bool = True

# Lớp model để quản lý trạng thái xử lý video
class VideoProcessingState:
    def __init__(self):
        self.last_detected_classes = set()
        
    def reset(self):
        self.last_detected_classes.clear()

# Khởi tạo trạng thái xử lý video
video_state = VideoProcessingState()

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

# Mô hình YOLO
class ModelManager:
    def __init__(self):
        self.model = None
        # Đường dẫn mặc định đến model YOLOv12 đã train
        self.model_path = os.path.join(root_dir, "models", "best_yolo12.pt")
        # Đường dẫn dự phòng nếu không tìm thấy model đã train
        self.backup_model_path = os.path.join(root_dir, "models", "yolo12n.pt")
        
    def load_model(self):
        if self.model is None:
            try:
                # Đầu tiên, tìm model YOLOv12 đã train
                if os.path.exists(self.model_path):
                    self.model = YOLO(self.model_path)
                    logger.info(f"Model loaded from {self.model_path}")
                # Nếu không tìm thấy, tìm trong thư mục models với model pretrained YOLOv12
                elif os.path.exists(self.backup_model_path):
                    self.model = YOLO(self.backup_model_path)
                    logger.info(f"Model loaded from {self.backup_model_path}")
                # Nếu không tìm thấy, tìm trong thư mục hiện tại
                elif os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")):
                    self.model = YOLO(os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt"))
                    logger.info("Model loaded from current directory")
                # Nếu không tìm thấy, dùng model mặc định yolo12n.pt
                else:
                    self.model = YOLO("yolo12n.pt")
                    logger.info("Using default YOLOv12n model")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                # Dùng model mặc định nếu có lỗi
                try:
                    self.model = YOLO("yolo12n.pt")
                    logger.info("Using default YOLOv12n model due to error")
                except Exception as e2:
                    logger.error(f"Error loading default YOLOv12n model: {e2}")
                    # Nếu vẫn lỗi, thử với YOLOv8n
                    self.model = YOLO("yolov8n.pt")
                    logger.info("Using YOLOv8n model as fallback")
        return self.model

# Khởi tạo quản lý mô hình và tải model ngay khi khởi tạo
model_manager = ModelManager()
model_manager.load_model()

# Hàm xóa các file tạm sau khi xử lý
def cleanup_temp_files(folder_path):
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            logger.info(f"Cleaned up temporary folder: {folder_path}")
    except Exception as e:
        logger.error(f"Error cleaning up temporary folder: {e}")

# Hàm tạo video từ các frame
async def create_video_from_frames(frames_folder, output_path, fps=30):
    try:
        # Sắp xếp files theo thứ tự số trong tên file
        def get_frame_number(filename):
            # Lấy số từ tên file, xử lý cả định dạng frame_X.jpg và frame_000001.jpg
            try:
                basename = os.path.basename(filename)
                # Lấy phần số từ tên file (loại bỏ 'frame_' và '.jpg')
                if '_' in basename:
                    num_part = basename.split('_')[1].split('.')[0]
                    return int(num_part)
                return 0
            except:
                return 0
        
        # Lấy tất cả file .jpg và sắp xếp theo số thứ tự
        frame_files = sorted(
            [os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith('.jpg')],
            key=get_frame_number
        )
        
        if not frame_files:
            logger.error("No frame files found")
            return None
            
        # Log number of frames và thứ tự
        logger.info(f"Processing {len(frame_files)} frames")
        if len(frame_files) > 0:
            logger.info(f"First frame: {os.path.basename(frame_files[0])}")
            if len(frame_files) > 1:
                logger.info(f"Second frame: {os.path.basename(frame_files[1])}")
            if len(frame_files) > 2:
                logger.info(f"Third frame: {os.path.basename(frame_files[2])}")
        
        # Đọc frame đầu tiên để xác định kích thước
        first_frame = cv2.imread(frame_files[0])
        if first_frame is None:
            logger.error(f"Could not read first frame: {frame_files[0]}")
            return None
            
        height, width, _ = first_frame.shape
        logger.info(f"Frame dimensions: {width}x{height}")
        
        # Tạo VideoWriter với chất lượng cao
        # Sử dụng codec H.264 hoặc fallback
        try:
            if os.name == 'nt':  # Windows
                fourcc = cv2.VideoWriter_fourcc(*'H264')
            else:  # Linux/Mac
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                
            video_writer = cv2.VideoWriter()
            video_writer.open(output_path, fourcc, fps, (width, height), True)
            
            # Nếu không thành công, thử các codec khác
            if not video_writer.isOpened():
                logger.info("Falling back to XVID codec")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                # Thử lại với mp4v nếu cần
                if not video_writer.isOpened():
                    logger.info("Falling back to mp4v codec")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
                    if not video_writer.isOpened():
                        logger.error("Failed to initialize any video codec")
                        return None
        except Exception as e:
            logger.error(f"Error initializing VideoWriter: {e}")
            # Fallback to mp4v
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Đếm frames được xử lý thành công
        processed_frames = 0
        
        # Ghi các frame vào video
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            if frame is not None:
                video_writer.write(frame)
                processed_frames += 1
            else:
                logger.error(f"Failed to read frame: {frame_file}")
        
        video_writer.release()
        logger.info(f"Successfully wrote {processed_frames} frames to video")
        
        # Kiểm tra xem file đã được tạo chưa
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            logger.error("Video was created but file is empty or not found")
            return None
    except Exception as e:
        logger.error(f"Error creating video: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# Endpoint để tạo video từ các frame
@app.post("/create-video")
async def create_video_endpoint(background_tasks: BackgroundTasks, request: Request, fps: float = Form(30.0)):
    try:
        # Tạo thư mục tạm để lưu các frame
        session_id = str(uuid.uuid4())
        frames_folder = os.path.join(temp_dir, f"frames_{session_id}")
        os.makedirs(frames_folder, exist_ok=True)
        
        # Lưu các frame vào thư mục tạm
        form = await request.form()
        frame_files = []
        
        # Log để debug
        logger.info(f"Received {len(form)} items in form data")
        
        # Kiểm tra cả hai định dạng có thể nhận được
        # Format 1: frame_data_X (base64 string)
        # Format 2: frame_X (file)
        frame_data_keys = [key for key in form.keys() if key.startswith('frame_data_')]
        frame_keys = [key for key in form.keys() if key.startswith('frame_') and not key.startswith('frame_data_') and key != 'frame_count']
        
        logger.info(f"Found {len(frame_data_keys)} frame_data keys and {len(frame_keys)} frame keys")
        
        # Nếu có dữ liệu frame_data (base64 strings)
        if frame_data_keys:
            logger.info("Processing frame_data (base64) format")
            # Sắp xếp keys để đảm bảo thứ tự frame đúng
            sorted_keys = sorted(frame_data_keys, key=lambda k: k.split('_')[-1])
            for key in sorted_keys:
                try:
                    base64_data = form[key]
                    # Nếu là chuỗi base64 đầy đủ với data:image/jpeg;base64,
                    if isinstance(base64_data, str) and base64_data.startswith('data:image/'):
                        # Lấy phần base64 thực sự (sau dấu phẩy)
                        base64_part = base64_data.split(',')[1]
                        # Chuyển base64 thành binary
                        binary_data = base64.b64decode(base64_part)
                    else:
                        # Nếu chỉ là chuỗi base64
                        binary_data = base64.b64decode(base64_data)
                    
                    # Lưu thành file, giữ nguyên định dạng tên để dễ sắp xếp
                    frame_path = os.path.join(frames_folder, f"frame_{key.split('_')[-1]}.jpg")
                    with open(frame_path, "wb") as f:
                        f.write(binary_data)
                    frame_files.append(frame_path)
                except Exception as e:
                    logger.error(f"Error processing frame data {key}: {e}")
        
        # Nếu có file frame
        elif frame_keys:
            logger.info("Processing frame (file upload) format")
            for key in sorted(frame_keys, key=lambda k: k.split('_')[-1]):
                file = form[key]
                if isinstance(file, UploadFile):
                    try:
                        frame_path = os.path.join(frames_folder, f"frame_{key.split('_')[-1]}.jpg")
                        contents = await file.read()
                        with open(frame_path, "wb") as f:
                            f.write(contents)
                        frame_files.append(frame_path)
                    except Exception as e:
                        logger.error(f"Error saving frame {key}: {e}")
        
        # Kiểm tra nếu có frame để xử lý
        if not frame_files:
            logger.error("No frames found after processing request data")
            return JSONResponse(
                status_code=400,
                content={"error": "No frames provided or no valid frames found"}
            )
        
        logger.info(f"Successfully saved {len(frame_files)} frames to {frames_folder}")
        
        # Đường dẫn cho video output
        output_video_path = os.path.join(temp_dir, f"processed_video_{session_id}.mp4")
        
        # Xác định FPS - sử dụng giá trị FPS do client gửi
        try:
            fps_value = float(fps)
            if fps_value < 1:
                fps_value = 30.0
            elif fps_value > 60:
                fps_value = 60.0
        except ValueError:
            fps_value = 30.0
        
        logger.info(f"Creating video with FPS: {fps_value}")
        
        # Tạo video từ các frame
        video_path = await create_video_from_frames(frames_folder, output_video_path, fps_value)
        
        if not video_path or not os.path.exists(video_path):
            logger.error("Failed to create video file or video path is None")
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to create video file"}
            )
        
        # Kiểm tra kích thước file video
        video_size = os.path.getsize(video_path)
        logger.info(f"Created video of size: {video_size} bytes")
        
        if video_size == 0:
            logger.error("Video file was created but is empty")
            return JSONResponse(
                status_code=500,
                content={"error": "Video file was created but is empty"}
            )
        
        # Đọc file video
        with open(video_path, "rb") as video_file:
            video_content = video_file.read()
        
        # Lên lịch xóa file tạm
        background_tasks.add_task(cleanup_temp_files, frames_folder)
        background_tasks.add_task(cleanup_temp_files, output_video_path)
        
        logger.info("Successfully created and returning video file")
        
        # Trả về video để tải xuống
        return StreamingResponse(
            io.BytesIO(video_content),
            media_type="video/mp4",
            headers={
                "Content-Disposition": "attachment;filename=processed_video.mp4"
            }
        )
    
    except Exception as e:
        logger.error(f"Error in create_video_endpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

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

# API xử lý frame video
@app.post("/process-video-frame")
async def process_video_frame(frame: VideoFrame):
    try:
        # Tải mô hình nếu chưa tải
        model = model_manager.load_model()
        
        # Chuyển base64 thành numpy array
        image_bytes = base64.b64decode(frame.image)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        # Kiểm tra nếu ảnh hợp lệ
        if image is None or image.size == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image data"}
            )
        
        # Giảm kích thước ảnh để tăng tốc độ xử lý nếu ảnh quá lớn
        height, width = image.shape[:2]
        max_dimension = 640  # Kích thước tối đa hợp lý cho xử lý nhanh
        
        if max(height, width) > max_dimension:
            # Tính toán tỷ lệ để giữ nguyên tỷ lệ khung hình
            scale = max_dimension / float(max(height, width))
            new_width = int(width * scale)
            new_height = int(height * scale)
            # Resize ảnh
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Thực hiện dự đoán - tối ưu tốc độ với cài đặt cao hơn
        results = model(image, conf=frame.conf, verbose=False, stream=True, iou=0.45)
        
        # Xử lý kết quả
        detections = []
        current_classes = set()
        
        # Lấy kết quả đầu tiên
        result = next(results)
        
        # Xử lý các box
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            boxes = result.boxes
            for box in boxes:
                # Lấy thông tin
                cls = int(box.cls[0])
                class_name = result.names[cls]
                conf_val = float(box.conf[0])
                
                # Thêm vào set lớp hiện tại
                current_classes.add(class_name)
                
                # Thêm vào danh sách
                detections.append({
                    "class": class_name,
                    "confidence": conf_val,
                    "description": class_descriptions.get(class_name, "")
                })
        
        # Vẽ kết quả - luôn vẽ bounding box
        annotated_frame = result.plot()
        
        # Nếu đã resize ảnh để xử lý, cần resize lại kết quả về kích thước gốc
        if max(height, width) > max_dimension:
            annotated_frame = cv2.resize(annotated_frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        
        # Chuyển kết quả thành base64 với chất lượng vừa đủ (85%) để cân bằng giữa chất lượng và tốc độ
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Trả về kết quả
        return {
            "image": result_base64,
            "detections": detections
        }
        
    except Exception as e:
        logger.error(f"Error processing video frame: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

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
    
    # Đảm bảo thư mục temp tồn tại
    os.makedirs(temp_dir, exist_ok=True)

# Dùng cho việc kiểm tra trạng thái API
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Chạy ứng dụng (khi test cục bộ)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 