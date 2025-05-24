import os
import sys

# Thêm thư mục cha vào path để có thể import app
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from app import app

# Export app cho Vercel
handler = app

# Đảm bảo app sẵn sàng
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 