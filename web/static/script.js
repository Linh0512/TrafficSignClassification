// Biến để theo dõi kết nối WebSocket
let ws;
let webcamStream;
let lastDetectedClasses = new Set();
let isWebcamRunning = false;

// Khi trang đã tải xong
document.addEventListener('DOMContentLoaded', function() {
    // Xử lý form upload ảnh
    const uploadForm = document.getElementById('upload-form');
    const confidenceSlider = document.getElementById('confidence');
    const confidenceValue = document.getElementById('conf-value');
    const resultImage = document.getElementById('result-image');
    const imagePlaceholder = document.getElementById('image-placeholder');
    const detectionResults = document.getElementById('detection-results');
    
    // Cập nhật giá trị hiển thị của ngưỡng tin cậy
    confidenceSlider.addEventListener('input', function() {
        confidenceValue.textContent = this.value;
    });
    
    // Xử lý sự kiện submit form
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const fileInput = document.getElementById('image-upload');
        const file = fileInput.files[0];
        
        if (!file) {
            alert('Vui lòng chọn một ảnh!');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('conf', confidenceSlider.value);
        
        // Hiển thị trạng thái đang xử lý
        const detectBtn = document.getElementById('detect-btn');
        detectBtn.disabled = true;
        detectBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Đang xử lý...';
        
        // Gửi yêu cầu đến API
        fetch('/detect-image', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Hiển thị ảnh kết quả
            resultImage.src = 'data:image/jpeg;base64,' + data.image;
            resultImage.classList.remove('d-none');
            imagePlaceholder.classList.add('d-none');
            
            // Hiển thị thông tin biển báo
            displayDetectionResults(data.detections, detectionResults);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Có lỗi xảy ra, vui lòng thử lại!');
        })
        .finally(() => {
            // Khôi phục trạng thái nút
            detectBtn.disabled = false;
            detectBtn.textContent = 'Nhận diện';
        });
    });
    
    // Xử lý Webcam
    const startWebcamBtn = document.getElementById('start-webcam');
    const stopWebcamBtn = document.getElementById('stop-webcam');
    const webcamConfidenceSlider = document.getElementById('webcam-confidence');
    const webcamConfidenceValue = document.getElementById('webcam-conf-value');
    const webcamResults = document.getElementById('webcam-detection-results');
    
    // Cập nhật giá trị hiển thị của ngưỡng tin cậy cho webcam
    webcamConfidenceSlider.addEventListener('input', function() {
        webcamConfidenceValue.textContent = this.value;
    });
    
    // Bắt đầu webcam
    startWebcamBtn.addEventListener('click', function() {
        startWebcam();
    });
    
    // Dừng webcam
    stopWebcamBtn.addEventListener('click', function() {
        stopWebcam();
    });
});

// Hàm hiển thị kết quả nhận diện
function displayDetectionResults(detections, container) {
    if (!detections || detections.length === 0) {
        container.innerHTML = '<p class="text-center">Không phát hiện biển báo nào trong ảnh!</p>';
        return;
    }
    
    let html = '';
    detections.forEach(detection => {
        html += `
        <div class="detected-sign">
            <h5>${detection.class}
                <span class="badge bg-primary confidence-badge">${Math.round(detection.confidence * 100)}%</span>
            </h5>
            <p>${detection.description || 'Không có mô tả'}</p>
        </div>
        `;
    });
    
    container.innerHTML = html;
}

// Hàm hiển thị kết quả nhận diện từ webcam - chỉ hiển thị lớp mới
function displayWebcamResults(detections, container) {
    if (!detections || detections.length === 0) {
        return;
    }
    
    // Tạo Set mới để lưu các lớp hiện tại
    const currentClasses = new Set();
    let html = '';
    let hasNewClass = false;
    
    detections.forEach(detection => {
        currentClasses.add(detection.class);
        
        // Chỉ hiển thị lớp mới
        if (!lastDetectedClasses.has(detection.class)) {
            hasNewClass = true;
            html += `
            <div class="detected-sign">
                <h5>${detection.class}
                    <span class="badge bg-primary confidence-badge">${Math.round(detection.confidence * 100)}%</span>
                </h5>
                <p>${detection.description || 'Không có mô tả'}</p>
            </div>
            `;
        }
    });
    
    // Nếu có lớp mới, cập nhật giao diện
    if (hasNewClass) {
        // Thêm vào đầu container thay vì thay thế
        container.innerHTML = html + container.innerHTML;
    }
    
    // Cập nhật danh sách lớp đã phát hiện
    lastDetectedClasses = currentClasses;
}

// Hàm bắt đầu webcam
function startWebcam() {
    const video = document.getElementById('webcam-video');
    const canvas = document.getElementById('webcam-canvas');
    const startBtn = document.getElementById('start-webcam');
    const stopBtn = document.getElementById('stop-webcam');
    const webcamResults = document.getElementById('webcam-detection-results');
    
    // Reset kết quả
    webcamResults.innerHTML = '<p class="text-center">Đang kết nối webcam...</p>';
    lastDetectedClasses.clear();
    
    // Kiểm tra hỗ trợ
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        webcamResults.innerHTML = '<p class="text-center text-danger">Trình duyệt của bạn không hỗ trợ webcam!</p>';
        return;
    }
    
    // Lấy stream từ webcam
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
            webcamStream = stream;
            video.srcObject = stream;
            
            // Cập nhật UI
            startBtn.classList.add('d-none');
            stopBtn.classList.remove('d-none');
            webcamResults.innerHTML = '<p class="text-center">Đang kết nối với máy chủ...</p>';
            
            // Kết nối WebSocket
            connectWebSocket();
        })
        .catch(function(err) {
            console.error('Error accessing webcam:', err);
            webcamResults.innerHTML = `<p class="text-center text-danger">Không thể truy cập webcam: ${err.message}</p>`;
        });
}

// Hàm dừng webcam
function stopWebcam() {
    const video = document.getElementById('webcam-video');
    const startBtn = document.getElementById('start-webcam');
    const stopBtn = document.getElementById('stop-webcam');
    
    // Dừng stream
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    
    // Đóng kết nối WebSocket
    if (ws) {
        ws.close();
        ws = null;
    }
    
    // Cập nhật UI
    video.srcObject = null;
    startBtn.classList.remove('d-none');
    stopBtn.classList.add('d-none');
    
    // Reset trạng thái
    isWebcamRunning = false;
}

// Hàm kết nối WebSocket
function connectWebSocket() {
    const video = document.getElementById('webcam-video');
    const canvas = document.getElementById('webcam-canvas');
    const ctx = canvas.getContext('2d');
    const webcamResults = document.getElementById('webcam-detection-results');
    const webcamConfidenceSlider = document.getElementById('webcam-confidence');
    
    // Xác định URL WebSocket
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = function() {
        console.log('WebSocket connection established');
        webcamResults.innerHTML = '<p class="text-center">Đã kết nối! Đang chờ phát hiện biển báo...</p>';
        isWebcamRunning = true;
        
        // Bắt đầu gửi frame
        sendWebcamFrames();
    };
    
    ws.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            
            // Hiển thị kết quả
            if (data.detections && data.detections.length > 0) {
                displayWebcamResults(data.detections, webcamResults);
            }
            
            // Hiển thị ảnh kết quả nếu có
            if (data.image) {
                const img = new Image();
                img.onload = function() {
                    canvas.width = img.width;
                    canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                };
                img.src = 'data:image/jpeg;base64,' + data.image;
            }
        } catch (e) {
            console.error('Error parsing WebSocket message:', e);
        }
    };
    
    ws.onclose = function() {
        console.log('WebSocket connection closed');
        if (isWebcamRunning) {
            webcamResults.innerHTML += '<p class="text-center text-warning">Kết nối bị đóng. Đang thử kết nối lại...</p>';
            setTimeout(connectWebSocket, 2000);
        }
    };
    
    ws.onerror = function(err) {
        console.error('WebSocket error:', err);
        webcamResults.innerHTML += '<p class="text-center text-danger">Lỗi kết nối WebSocket!</p>';
    };
    
    // Hàm để gửi frame từ webcam
    function sendWebcamFrames() {
        if (!isWebcamRunning || !ws || ws.readyState !== WebSocket.OPEN) {
            return;
        }
        
        // Điều chỉnh kích thước canvas
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Vẽ frame hiện tại lên canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Chuyển canvas thành base64
        const imageData = canvas.toDataURL('image/jpeg', 0.7);
        
        // Gửi dữ liệu qua WebSocket
        ws.send(JSON.stringify({
            image: imageData.split(',')[1],
            conf: webcamConfidenceSlider.value
        }));
        
        // Lên lịch gửi frame tiếp theo (khoảng 10 FPS)
        setTimeout(sendWebcamFrames, 100);
    }
} 