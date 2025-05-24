import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import yaml
from collections import Counter
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import sys

# Thêm thư mục gốc vào sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, root_dir)

# Đường dẫn đến thư mục dataset
DATASET_DIR = os.path.join(root_dir, 'data')
# Đường dẫn để lưu kết quả phân tích
EDA_RESULTS_DIR = os.path.join(root_dir, 'outputs', 'eda_results')

def load_yaml_config():
    """Tải file cấu hình YAML của dataset."""
    with open(os.path.join(DATASET_DIR, 'data.yaml'), 'r') as f:
        return yaml.safe_load(f)

def count_classes_in_labels(folder_path):
    """Đếm số lượng đối tượng theo từng loại class trong thư mục nhãn."""
    class_counts = Counter()
    
    # Liệt kê tất cả các file nhãn
    label_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    for label_file in tqdm(label_files, desc=f"Đang phân tích {folder_path}"):
        file_path = os.path.join(folder_path, label_file)
        
        # Đọc file và đếm các class
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():  # Đảm bảo dòng không trống
                        class_id = int(line.strip().split()[0])
                        class_counts[class_id] += 1
        except Exception as e:
            print(f"Lỗi khi đọc file {file_path}: {e}")
    
    return class_counts

def analyze_image_sizes(folder_path):
    """Phân tích kích thước của các hình ảnh trong dataset."""
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    widths, heights = [], []
    
    for img_file in tqdm(image_files, desc=f"Đang phân tích kích thước ảnh trong {folder_path}"):
        img_path = os.path.join(folder_path, img_file)
        try:
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            heights.append(h)
            widths.append(w)
        except Exception as e:
            print(f"Lỗi khi đọc ảnh {img_path}: {e}")
    
    return widths, heights

def analyze_bbox_sizes(folder_path):
    """Phân tích kích thước các bounding box trong dataset."""
    label_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    relative_widths, relative_heights = [], []
    classes = []
    
    for label_file in tqdm(label_files, desc=f"Đang phân tích bounding box trong {folder_path}"):
        file_path = os.path.join(folder_path, label_file)
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) == 5:  # Format YOLO: class x_center y_center width height
                            class_id = int(parts[0])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            classes.append(class_id)
                            relative_widths.append(width)
                            relative_heights.append(height)
        except Exception as e:
            print(f"Lỗi khi đọc file {file_path}: {e}")
    
    return classes, relative_widths, relative_heights

def plot_class_distribution(class_counts, class_names, title):
    """Vẽ biểu đồ phân phối lớp."""
    plt.figure(figsize=(15, 8))
    
    # Chuyển đổi class_id sang tên class
    labels = [class_names.get(str(class_id), f"Class {class_id}") for class_id in class_counts.keys()]
    counts = list(class_counts.values())
    
    # Sắp xếp theo số lượng giảm dần
    sorted_indices = np.argsort(counts)[::-1]
    labels = [labels[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    
    # Vẽ biểu đồ cột
    plt.bar(range(len(counts)), counts)
    plt.xticks(range(len(counts)), labels, rotation=90)
    plt.xlabel('Class')
    plt.ylabel('Số lượng')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.show()

def plot_image_size_distribution(widths, heights):
    """Vẽ biểu đồ phân phối kích thước ảnh."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=30)
    plt.xlabel('Chiều rộng (pixels)')
    plt.ylabel('Số lượng')
    plt.title('Phân phối chiều rộng ảnh')
    
    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=30)
    plt.xlabel('Chiều cao (pixels)')
    plt.ylabel('Số lượng')
    plt.title('Phân phối chiều cao ảnh')
    
    plt.tight_layout()
    plt.savefig('image_size_distribution.png')
    plt.show()
    
    # Vẽ biểu đồ phân tán
    plt.figure(figsize=(8, 8))
    plt.scatter(widths, heights, alpha=0.5)
    plt.xlabel('Chiều rộng (pixels)')
    plt.ylabel('Chiều cao (pixels)')
    plt.title('Phân phối kích thước ảnh')
    plt.tight_layout()
    plt.savefig('image_size_scatter.png')
    plt.show()

def plot_bbox_size_distribution(classes, widths, heights, class_names):
    """Vẽ biểu đồ phân phối kích thước bounding box."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=30)
    plt.xlabel('Chiều rộng tương đối')
    plt.ylabel('Số lượng')
    plt.title('Phân phối chiều rộng bounding box')
    
    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=30)
    plt.xlabel('Chiều cao tương đối')
    plt.ylabel('Số lượng')
    plt.title('Phân phối chiều cao bounding box')
    
    plt.tight_layout()
    plt.savefig('bbox_size_distribution.png')
    plt.show()
    
    # Vẽ biểu đồ phân tán
    plt.figure(figsize=(10, 8))
    
    # Giới hạn số lượng classes được hiển thị để biểu đồ dễ đọc hơn
    top_classes = Counter(classes).most_common(10)
    top_class_ids = [class_id for class_id, _ in top_classes]
    
    for class_id in top_class_ids:
        indices = [i for i, c in enumerate(classes) if c == class_id]
        class_widths = [widths[i] for i in indices]
        class_heights = [heights[i] for i in indices]
        plt.scatter(class_widths, class_heights, alpha=0.5, label=class_names.get(str(class_id), f"Class {class_id}"))
    
    plt.xlabel('Chiều rộng tương đối')
    plt.ylabel('Chiều cao tương đối')
    plt.title('Phân phối kích thước bounding box theo class')
    plt.legend()
    plt.tight_layout()
    plt.savefig('bbox_size_by_class.png')
    plt.show()

def visualize_random_samples(image_folder, label_folder, class_names, num_samples=5):
    """Hiển thị ngẫu nhiên một số mẫu từ dataset với bounding box."""
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        print(f"Không tìm thấy ảnh trong {image_folder}")
        return
    
    # Chọn ngẫu nhiên các file ảnh
    selected_files = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
    
    plt.figure(figsize=(15, 12))
    for i, img_file in enumerate(selected_files):
        img_path = os.path.join(image_folder, img_file)
        label_file = os.path.join(label_folder, os.path.splitext(img_file)[0] + '.txt')
        
        # Đọc ảnh
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w, _ = img.shape
        
        # Đọc nhãn
        boxes = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1]) * w
                            y_center = float(parts[2]) * h
                            width = float(parts[3]) * w
                            height = float(parts[4]) * h
                            
                            x1 = int(x_center - width / 2)
                            y1 = int(y_center - height / 2)
                            x2 = int(x_center + width / 2)
                            y2 = int(y_center + height / 2)
                            
                            boxes.append((class_id, x1, y1, x2, y2))
        
        # Hiển thị ảnh với bounding box
        plt.subplot(np.ceil(num_samples / 2).astype(int), 2, i + 1)
        plt.imshow(img)
        
        for box in boxes:
            class_id, x1, y1, x2, y2 = box
            class_name = class_names.get(str(class_id), f"Class {class_id}")
            
            # Vẽ bounding box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
            
            # Thêm label
            plt.text(x1, y1 - 5, class_name, color='white', backgroundcolor='red', fontsize=8)
        
        plt.title(f"Sample {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('random_samples.png')
    plt.show()

def main():
    # Tạo thư mục để lưu kết quả EDA
    os.makedirs(EDA_RESULTS_DIR, exist_ok=True)
    
    # Tải cấu hình
    config = load_yaml_config()
    print("Thông tin dataset:")
    print(f"Số lượng classes: {config['nc']}")
    print(f"Tên các classes: {config['names']}")
    
    # Tạo dictionary ánh xạ từ class_id sang tên
    class_names = {str(i): name for i, name in enumerate(config['names'])}
    
    # Phân tích phân phối class trong tập huấn luyện
    train_class_counts = count_classes_in_labels(os.path.join(DATASET_DIR, 'train', 'labels'))
    test_class_counts = count_classes_in_labels(os.path.join(DATASET_DIR, 'test', 'labels'))
    valid_class_counts = count_classes_in_labels(os.path.join(DATASET_DIR, 'valid', 'labels'))
    
    # In một số thống kê cơ bản
    print("\nThống kê tập huấn luyện:")
    print(f"Tổng số đối tượng: {sum(train_class_counts.values())}")
    print(f"Số lượng classes có trong tập huấn luyện: {len(train_class_counts)}")
    
    print("\nThống kê tập kiểm thử:")
    print(f"Tổng số đối tượng: {sum(test_class_counts.values())}")
    print(f"Số lượng classes có trong tập kiểm thử: {len(test_class_counts)}")
    
    print("\nThống kê tập validation:")
    print(f"Tổng số đối tượng: {sum(valid_class_counts.values())}")
    print(f"Số lượng classes có trong tập validation: {len(valid_class_counts)}")
    
    # Vẽ biểu đồ phân phối các lớp
    plot_class_distribution(train_class_counts, class_names, "Phân phối lớp trong tập huấn luyện")
    plot_class_distribution(test_class_counts, class_names, "Phân phối lớp trong tập kiểm thử")
    plot_class_distribution(valid_class_counts, class_names, "Phân phối lớp trong tập validation")
    
    # Phân tích kích thước ảnh
    train_widths, train_heights = analyze_image_sizes(os.path.join(DATASET_DIR, 'train', 'images'))
    plot_image_size_distribution(train_widths, train_heights)
    
    # Phân tích kích thước bounding box
    classes, bbox_widths, bbox_heights = analyze_bbox_sizes(os.path.join(DATASET_DIR, 'train', 'labels'))
    plot_bbox_size_distribution(classes, bbox_widths, bbox_heights, class_names)
    
    # Hiển thị một số mẫu ngẫu nhiên từ tập huấn luyện
    visualize_random_samples(
        os.path.join(DATASET_DIR, 'train', 'images'),
        os.path.join(DATASET_DIR, 'train', 'labels'),
        class_names
    )
    
    # Tạo báo cáo tổng quan dưới dạng DataFrame
    class_stats = []
    for class_id in range(config['nc']):
        class_name = class_names.get(str(class_id), f"Class {class_id}")
        train_count = train_class_counts.get(class_id, 0)
        valid_count = valid_class_counts.get(class_id, 0)
        test_count = test_class_counts.get(class_id, 0)
        total_count = train_count + valid_count + test_count
        
        class_stats.append({
            'class_id': class_id,
            'class_name': class_name,
            'train_count': train_count,
            'valid_count': valid_count,
            'test_count': test_count,
            'total_count': total_count
        })
    
    stats_df = pd.DataFrame(class_stats)
    stats_df = stats_df.sort_values('total_count', ascending=False)
    
    print("\nThống kê chi tiết theo class:")
    print(stats_df)
    
    # Lưu vào file CSV
    stats_df.to_csv(os.path.join(EDA_RESULTS_DIR, 'class_statistics.csv'), index=False)
    
    # Hiển thị phân phối đối tượng giữa các tập
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=stats_df.head(20),  # Chỉ hiển thị 20 lớp hàng đầu
        x='class_name', 
        y='total_count'
    )
    plt.xticks(rotation=90)
    plt.title('Tổng số đối tượng theo class (20 lớp hàng đầu)')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_RESULTS_DIR, 'top_classes.png'))
    plt.show()

if __name__ == "__main__":
    main() 