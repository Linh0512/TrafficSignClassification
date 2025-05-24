// Biến để theo dõi kết nối WebSocket
let ws;
let webcamStream;
let lastDetectedClasses = new Set();
let isWebcamRunning = false;

// Biến cho xử lý video
let videoProcessor = {
    videoElement: null,
    canvasElement: null,
    ctx: null,
    isProcessing: false,
    videoSource: null,
    lastVideoDetectedClasses: new Set(),
    frameRate: 30, // Tăng từ 24 FPS lên 30 FPS
    frameInterval: 1000 / 30, // ms per frame (30 FPS)
    confidenceThreshold: 0.25,
    stabilize: true,
    processTimer: null,
    detectionDelay: 0, // để theo dõi thời gian xử lý
    frameCount: 0,
    lastFpsUpdateTime: 0,
    actualFps: 0,
    pendingRequest: false, // Cờ để kiểm soát trùng lặp request
    processedFrames: [], // Mảng lưu trữ các frame đã xử lý
    isRecording: false, // Trạng thái đang ghi
    controlsVisible: false, // Trạng thái hiển thị controls
    processingPlaceholder: null, // Phần tử hiển thị trạng thái đang xử lý
    
    // Khởi tạo trình xử lý video
    init: function(videoElementId, canvasElementId) {
        this.videoElement = document.getElementById(videoElementId);
        this.canvasElement = document.getElementById(canvasElementId);
        this.ctx = this.canvasElement.getContext('2d');
        
        // Đăng ký sự kiện kết thúc video
        this.videoElement.addEventListener('ended', () => {
            this.stop();
            // Hiển thị nút tải xuống nếu có frames đã xử lý
            if (this.processedFrames.length > 0) {
                this.showDownloadButton();
            }
        });
        
        // Đảm bảo video hiển thị đúng kích thước
        this.videoElement.addEventListener('loadedmetadata', () => {
            this.updateCanvasSize();
        });
        
        // Thêm nút tải xuống video
        this.createDownloadButton();
        
        // Tạo placeholder hiển thị khi đang xử lý
        this.createProcessingPlaceholder();
        
        // Thêm sự kiện để hiển thị controls khi hover
        const videoWrapper = document.querySelector('.video-wrapper');
        if (videoWrapper) {
            videoWrapper.addEventListener('mouseenter', () => {
                if (this.isProcessing) {
                    this.showControls();
                }
            });
            
            videoWrapper.addEventListener('mouseleave', () => {
                if (this.isProcessing) {
                    this.hideControls();
                }
            });
            
            // Thêm sự kiện click để hiển thị/ẩn controls
            videoWrapper.addEventListener('click', (e) => {
                // Ngăn click vào canvas truyền xuống video
                if (e.target === this.canvasElement) {
                    e.preventDefault();
                    this.toggleControls();
                    
                    // Truyền sự kiện click xuống video
                    const clickEvent = new MouseEvent('click', {
                        bubbles: true,
                        cancelable: true,
                        view: window
                    });
                    this.videoElement.dispatchEvent(clickEvent);
                }
            });
        }
    },
    
    // Tạo phần tử hiển thị khi đang xử lý
    createProcessingPlaceholder: function() {
        // Tạo phần tử processing placeholder
        this.processingPlaceholder = document.createElement('div');
        this.processingPlaceholder.className = 'processing-placeholder text-center';
        this.processingPlaceholder.innerHTML = `
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Đang xử lý...</span>
            </div>
            <p class="mt-2">Đang xử lý video, vui lòng đợi...</p>
        `;
        
        // Thêm style cho placeholder
        this.processingPlaceholder.style.position = 'absolute';
        this.processingPlaceholder.style.top = '50%';
        this.processingPlaceholder.style.left = '50%';
        this.processingPlaceholder.style.transform = 'translate(-50%, -50%)';
        this.processingPlaceholder.style.backgroundColor = 'rgba(255, 255, 255, 0.8)';
        this.processingPlaceholder.style.padding = '20px';
        this.processingPlaceholder.style.borderRadius = '8px';
        this.processingPlaceholder.style.display = 'none';
        
        // Thêm vào video-wrapper
        const videoWrapper = document.querySelector('.video-wrapper');
        if (videoWrapper) {
            videoWrapper.appendChild(this.processingPlaceholder);
        }
    },
    
    // Hiển thị controls của video
    showControls: function() {
        if (this.videoElement) {
            this.videoElement.controls = true;
            this.controlsVisible = true;
            
            // Đảm bảo z-index của canvas thấp hơn 
            this.canvasElement.style.zIndex = "5";
            this.canvasElement.style.pointerEvents = "none";
        }
    },
    
    // Ẩn controls của video
    hideControls: function() {
        if (this.videoElement && !this.videoElement.paused) {
            this.videoElement.controls = false;
            this.controlsVisible = false;
            
            // Phục hồi z-index của canvas
            this.canvasElement.style.zIndex = "10";
            this.canvasElement.style.pointerEvents = "auto";
        }
    },
    
    // Chuyển đổi trạng thái controls
    toggleControls: function() {
        if (this.controlsVisible) {
            this.hideControls();
        } else {
            this.showControls();
        }
    },
    
    // Tạo nút tải xuống video
    createDownloadButton: function() {
        const downloadBtnContainer = document.createElement('div');
        downloadBtnContainer.className = 'mt-3';
        downloadBtnContainer.id = 'download-container';
        downloadBtnContainer.style.display = 'none';
        
        const downloadBtn = document.createElement('button');
        downloadBtn.className = 'btn btn-success';
        downloadBtn.id = 'download-video-btn';
        downloadBtn.textContent = 'Tải xuống video đã xử lý';
        downloadBtn.addEventListener('click', () => this.downloadProcessedVideo());
        
        downloadBtnContainer.appendChild(downloadBtn);
        
        // Thêm vào sau video-result-container
        const videoResultContainer = document.getElementById('video-result-container');
        videoResultContainer.parentNode.insertBefore(downloadBtnContainer, videoResultContainer.nextSibling);
    },
    
    // Hiển thị nút tải xuống
    showDownloadButton: function() {
        const downloadContainer = document.getElementById('download-container');
        if (downloadContainer) {
            downloadContainer.style.display = 'block';
        }
    },
    
    // Tải xuống video đã xử lý
    downloadProcessedVideo: function() {
        if (this.processedFrames.length === 0) {
            alert('Không có frame nào được xử lý để tạo video!');
            return;
        }
        
        // Hiển thị thông báo đang xử lý
        alert('Video đang được xử lý và sẽ được tải xuống khi hoàn thành. Vui lòng đợi...');
        
        try {
            // Sử dụng phương pháp gửi data - gửi base64 strings
            const formData = new FormData();
            
            // Thêm thông tin về số lượng frames
            formData.append('frame_count', this.processedFrames.length);
            
            console.log(`Sending all ${this.processedFrames.length} frames to maintain original video timing`);
            
            // Thêm TẤT CẢ frames vào FormData - không bỏ bớt frame nào
            for (let i = 0; i < this.processedFrames.length; i++) {
                // Đảm bảo thứ tự frame chính xác bằng cách sử dụng số thứ tự có padding zeros
                // Ví dụ: frame_000001, frame_000002, ...
                const paddedIndex = String(i).padStart(6, '0');
                formData.append(`frame_data_${paddedIndex}`, this.processedFrames[i]);
            }
            
            // Thêm thông tin FPS - sử dụng FPS của video gốc
            // Nếu actualFps = 0, sử dụng frameRate mặc định
            const originalFps = this.actualFps || this.frameRate;
            formData.append('fps', originalFps);
            
            console.log(`Using original FPS: ${originalFps}`);
            
            // Gửi request để tạo video
            fetch('/create-video', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    return response.text().then(text => {
                        try {
                            const data = JSON.parse(text);
                            throw new Error(data.error || 'Có lỗi khi tạo video');
                        } catch (e) {
                            throw new Error('Có lỗi khi tạo video: ' + text);
                        }
                    });
                }
                return response.blob();
            })
            .then(blob => {
                // Tạo URL để tải xuống
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'processed_video.mp4';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            })
            .catch(error => {
                console.error('Error creating video:', error);
                alert('Có lỗi khi tạo video: ' + error.message);
            });
        } catch (error) {
            console.error('Error preparing frames for video:', error);
            alert('Có lỗi khi chuẩn bị frames cho video: ' + error.message);
        }
    },
    
    // Chuẩn bị video để xử lý
    prepareVideo: function(file, confidenceThreshold, stabilize) {
        // Reset trạng thái
        this.stop();
        this.confidenceThreshold = confidenceThreshold;
        this.stabilize = stabilize;
        this.lastVideoDetectedClasses.clear();
        this.pendingRequest = false;
        this.processedFrames = []; // Reset mảng frame đã xử lý
        
        // Ẩn nút tải xuống
        const downloadContainer = document.getElementById('download-container');
        if (downloadContainer) {
            downloadContainer.style.display = 'none';
        }
        
        // Tạo URL cho video
        if (this.videoSource) {
            URL.revokeObjectURL(this.videoSource);
        }
        this.videoSource = URL.createObjectURL(file);
        this.videoElement.src = this.videoSource;
        
        // Ẩn video placeholder
        document.getElementById('video-placeholder').classList.add('d-none');
        
        // Chờ metadata để lấy thông tin video
        this.videoElement.onloadedmetadata = () => {
            console.log('Video loaded, dimensions:', this.videoElement.videoWidth, 'x', this.videoElement.videoHeight);
            
            // Cập nhật kích thước canvas
            this.updateCanvasSize();
            
            // Thêm sự kiện resize
            window.addEventListener('resize', () => this.updateCanvasSize());
        };
        
        // Thêm sự kiện khi video đã tải
        this.videoElement.onloadeddata = () => {
            this.updateCanvasSize();
        };
    },
    
    // Bắt đầu xử lý video
    start: function() {
        if (!this.videoElement || !this.videoSource) return;
        
        this.isProcessing = true;
        
        // Chuẩn bị hiển thị
        const videoResultContainer = document.getElementById('video-result-container');
        videoResultContainer.classList.add('processing');
        
        // Ẩn video và canvas ban đầu, chỉ hiển thị placeholder "đang xử lý"
        this.videoElement.classList.add('d-none');
        this.canvasElement.classList.add('d-none');
        
        // Hiển thị placeholder "đang xử lý"
        if (this.processingPlaceholder) {
            this.processingPlaceholder.style.display = 'block';
        }
        
        // Bắt đầu phát video nhưng ẩn đi
        this.videoElement.play();
        
        // Điều chỉnh kích thước canvas cho phù hợp với kích thước video
        this.updateCanvasSize();
        
        // Cập nhật giao diện
        const detectBtn = document.getElementById('video-detect-btn');
        const stopBtn = document.getElementById('video-stop-btn');
        detectBtn.classList.add('d-none');
        stopBtn.classList.remove('d-none');
        
        // Cập nhật trạng thái xử lý
        document.getElementById('video-processing-status').textContent = 'Đang xử lý';
        
        // Reset các biến theo dõi FPS
        this.frameCount = 0;
        this.lastFpsUpdateTime = performance.now();
        this.actualFps = 0;
        this.pendingRequest = false;
        this.isRecording = true;
        
        // Bắt đầu xử lý frame với FPS cao hơn
        this.processFrame();
    },
    
    // Dừng xử lý video
    stop: function() {
        this.isProcessing = false;
        this.isRecording = false;
        
        if (this.videoElement) {
            this.videoElement.pause();
        }
        
        // Hủy timer
        if (this.processTimer) {
            clearTimeout(this.processTimer);
            this.processTimer = null;
        }
        
        // Ẩn placeholder "đang xử lý"
        if (this.processingPlaceholder) {
            this.processingPlaceholder.style.display = 'none';
        }
        
        // Cập nhật giao diện
        const detectBtn = document.getElementById('video-detect-btn');
        const stopBtn = document.getElementById('video-stop-btn');
        if (detectBtn && stopBtn) {
            detectBtn.classList.remove('d-none');
            stopBtn.classList.add('d-none');
        }
        
        // Gỡ bỏ class processing
        const videoResultContainer = document.getElementById('video-result-container');
        videoResultContainer.classList.remove('processing');
        
        // Cập nhật trạng thái xử lý
        document.getElementById('video-processing-status').textContent = 'Không';
        
        // Hiển thị nút download nếu có frame đã xử lý
        if (this.processedFrames.length > 0) {
            this.showDownloadButton();
        }
    },
    
    // Cập nhật kích thước canvas
    updateCanvasSize: function() {
        if (!this.videoElement || !this.canvasElement) return;
        
        // Đảm bảo canvas có cùng kích thước với video
        if (this.videoElement.videoWidth && this.videoElement.videoHeight) {
            // Cập nhật kích thước thực của canvas để vẽ đúng
            this.canvasElement.width = this.videoElement.videoWidth;
            this.canvasElement.height = this.videoElement.videoHeight;
            
            console.log('Canvas dimensions set to:', this.canvasElement.width, 'x', this.canvasElement.height);
        }
    },
    
    // Xử lý từng frame của video
    processFrame: function() {
        if (!this.isProcessing || this.videoElement.paused || this.videoElement.ended) {
            return;
        }
        
        // Nếu đang có request đang chờ xử lý, bỏ qua frame này để tránh tắc nghẽn
        if (this.pendingRequest) {
            this.processTimer = setTimeout(() => this.processFrame(), this.frameInterval);
            return;
        }
        
        // Tính toán thời gian cho mỗi frame để đạt được frameRate mong muốn
        const startTime = performance.now();
        
        // Đảm bảo canvas có đúng kích thước trước khi vẽ
        if (this.canvasElement.width !== this.videoElement.videoWidth ||
            this.canvasElement.height !== this.videoElement.videoHeight) {
            this.canvasElement.width = this.videoElement.videoWidth;
            this.canvasElement.height = this.videoElement.videoHeight;
            console.log('Updated canvas dimensions:', this.canvasElement.width, 'x', this.canvasElement.height);
        }
        
        // Vẽ frame hiện tại vào canvas (không hiển thị) để chuẩn bị gửi đến server
        this.ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
        this.ctx.drawImage(
            this.videoElement,
            0, 0, this.videoElement.videoWidth, this.videoElement.videoHeight,
            0, 0, this.canvasElement.width, this.canvasElement.height
        );
        
        // Chuyển canvas thành base64 với chất lượng cao hơn để tăng chất lượng
        const imageData = this.canvasElement.toDataURL('image/jpeg', 0.85);
        
        // Đánh dấu đang có request đang chờ xử lý
        this.pendingRequest = true;
        
        // Gửi frame đến server để xử lý
        this.sendFrameToServer(imageData.split(',')[1]);
        
        // Cập nhật FPS mỗi giây
        this.frameCount++;
        const now = performance.now();
        const elapsed = now - this.lastFpsUpdateTime;
        
        if (elapsed >= 1000) { // Cập nhật FPS mỗi giây
            this.actualFps = Math.round((this.frameCount * 1000) / elapsed);
            document.getElementById('video-fps').textContent = this.actualFps;
            this.frameCount = 0;
            this.lastFpsUpdateTime = now;
        }
        
        // Tính toán thời gian xử lý
        const processingTime = performance.now() - startTime;
        
        // Cập nhật detectionDelay
        this.detectionDelay = processingTime;
        
        // Tính toán thời gian chờ để đạt được frameRate - giảm thời gian chờ tối thiểu
        const waitTime = Math.max(1, this.frameInterval - processingTime);
        
        // Lên lịch xử lý frame tiếp theo, tính toán cân bằng
        this.processTimer = setTimeout(() => this.processFrame(), waitTime);
    },
    
    // Gửi frame đến server qua fetch API
    sendFrameToServer: function(base64Data) {
        fetch('/process-video-frame', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: base64Data,
                conf: this.confidenceThreshold,
                stabilize: this.stabilize
            }),
            priority: 'high' // Ưu tiên cao cho request
        })
        .then(response => response.json())
        .then(data => {
            // Đánh dấu đã xử lý xong request
            this.pendingRequest = false;
            
            // Xử lý kết quả từ server
            this.handleDetectionResult(data);
            
            // Lưu frame đã xử lý để tạo video nếu đang ghi
            if (this.isRecording && data.image) {
                this.processedFrames.push('data:image/jpeg;base64,' + data.image);
            }
        })
        .catch(error => {
            // Đánh dấu đã xử lý xong request ngay cả khi có lỗi
            this.pendingRequest = false;
            console.error('Error processing video frame:', error);
        });
    },
    
    // Xử lý kết quả nhận diện
    handleDetectionResult: function(data) {
        const videoDetectionResults = document.getElementById('video-detection-results');
        
        // Hiển thị frame được xử lý với bounding box
        if (data.image) {
            // Ẩn placeholder "đang xử lý" khi đã có kết quả đầu tiên
            if (this.processingPlaceholder) {
                this.processingPlaceholder.style.display = 'none';
            }
            
            // Hiển thị canvas với kết quả xử lý
            this.canvasElement.classList.remove('d-none');
            
            const img = new Image();
            img.onload = () => {
                // Xóa canvas trước khi vẽ
                this.ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
                
                // Vẽ hình ảnh đã xử lý lên canvas với đúng kích thước
                this.ctx.drawImage(
                    img,
                    0, 0, img.width, img.height,
                    0, 0, this.canvasElement.width, this.canvasElement.height
                );
            };
            img.src = 'data:image/jpeg;base64,' + data.image;
        }
        
        // Xử lý và hiển thị thông tin nhận diện
        if (data.detections && data.detections.length > 0) {
            // Nếu bật chế độ ổn định, chỉ hiển thị các lớp mới
            if (this.stabilize) {
                // Lọc các lớp mới
                const currentClasses = new Set();
                let newDetections = [];
                
                data.detections.forEach(detection => {
                    currentClasses.add(detection.class);
                    if (!this.lastVideoDetectedClasses.has(detection.class)) {
                        newDetections.push(detection);
                    }
                });
                
                // Chỉ cập nhật giao diện nếu có phát hiện mới
                if (newDetections.length > 0) {
                    let html = '';
                    newDetections.forEach(detection => {
                        html += `
                        <div class="detected-sign">
                            <h5>${detection.class}
                                <span class="badge bg-primary confidence-badge">${Math.round(detection.confidence * 100)}%</span>
                            </h5>
                            <p>${detection.description || 'Không có mô tả'}</p>
                        </div>
                        `;
                    });
                    
                    // Thêm vào đầu danh sách hiện tại
                    videoDetectionResults.innerHTML = html + videoDetectionResults.innerHTML;
                }
                
                // Cập nhật danh sách lớp đã phát hiện
                this.lastVideoDetectedClasses = currentClasses;
                
                // Reset danh sách nếu không có phát hiện nào
                if (data.detections.length === 0) {
                    this.lastVideoDetectedClasses.clear();
                }
            } else {
                // Hiển thị tất cả phát hiện nếu không bật chế độ ổn định
                let html = '';
                data.detections.forEach(detection => {
                    html += `
                    <div class="detected-sign">
                        <h5>${detection.class}
                            <span class="badge bg-primary confidence-badge">${Math.round(detection.confidence * 100)}%</span>
                        </h5>
                        <p>${detection.description || 'Không có mô tả'}</p>
                    </div>
                    `;
                });
                
                videoDetectionResults.innerHTML = html;
            }
        }
    }
};

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
    
    // Xử lý Video
    const videoUploadForm = document.getElementById('video-upload-form');
    const videoConfidenceSlider = document.getElementById('video-confidence');
    const videoConfidenceValue = document.getElementById('video-conf-value');
    const videoStabilizeCheckbox = document.getElementById('video-stabilize');
    const videoStopBtn = document.getElementById('video-stop-btn');
    
    // Khởi tạo trình xử lý video
    videoProcessor.init('video-player', 'video-canvas');
    
    // Cập nhật giá trị hiển thị của ngưỡng tin cậy cho video
    videoConfidenceSlider.addEventListener('input', function() {
        videoConfidenceValue.textContent = this.value;
    });
    
    // Xử lý sự kiện dừng video
    videoStopBtn.addEventListener('click', function() {
        videoProcessor.stop();
    });
    
    // Xử lý sự kiện submit form video
    videoUploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const fileInput = document.getElementById('video-upload');
        const file = fileInput.files[0];
        
        if (!file) {
            alert('Vui lòng chọn một video!');
            return;
        }
        
        // Chuẩn bị video để xử lý
        videoProcessor.prepareVideo(
            file, 
            videoConfidenceSlider.value,
            videoStabilizeCheckbox.checked
        );
        
        // Reset kết quả nhận diện
        document.getElementById('video-detection-results').innerHTML = 
            '<p class="text-center">Đang xử lý video, kết quả sẽ hiển thị ở đây...</p>';
        
        // Bắt đầu xử lý video
        videoProcessor.start();
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
    
    // Xử lý tab thông tin mô hình
    const infoTab = document.getElementById('info-tab');
    const modelDetails = document.getElementById('model-details');
    
    // Tải thông tin mô hình khi tab được mở
    infoTab.addEventListener('shown.bs.tab', function (e) {
        loadModelInfo();
    });
    
    // Tải thông tin mô hình
    function loadModelInfo() {
        fetch('/model-info')
            .then(response => response.json())
            .then(data => {
                const modelPath = data.model;
                const classes = data.classes;
                
                // Tạo HTML hiển thị thông tin mô hình
                let html = `
                    <div class="model-info-card">
                        <div class="model-path">
                            <strong>Đường dẫn mô hình:</strong> ${modelPath}
                        </div>
                        <div class="model-classes mt-3">
                            <strong>Số lượng lớp:</strong> ${Object.keys(classes).length}
                        </div>
                        <div class="mt-3">
                            <strong>Các lớp nhận diện:</strong>
                            <div class="class-list mt-2">
                `;
                
                // Tạo danh sách các lớp
                Object.entries(classes).forEach(([id, name]) => {
                    html += `<span class="badge bg-primary me-2 mb-2">${name}</span>`;
                });
                
                html += `
                            </div>
                        </div>
                        
                        <div class="mt-4">
                            <h6>Thông tin hiệu suất:</h6>
                            <p>Mô hình YOLOv12 đạt mAP50 = 0.78 và mAP50-95 = 0.658 trên tập dữ liệu kiểm định.</p>
                            <p><em>Thông tin này dựa trên kết quả đánh giá gần nhất của mô hình.</em></p>
                        </div>
                    </div>
                `;
                
                modelDetails.innerHTML = html;
            })
            .catch(error => {
                console.error('Error fetching model info:', error);
                modelDetails.innerHTML = `
                    <div class="alert alert-danger">
                        Không thể tải thông tin mô hình. Vui lòng thử lại sau.
                    </div>
                `;
            });
    }
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