# Import thư viện cần thiết
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Tải mô hình YOLOv8 pre-trained
model = YOLO(r"checkpoint\My_model\best.pt")  

# Đọc ảnh từ file
image_path = r'image.png'
img = cv2.imread(image_path)

# Chuyển đổi từ BGR (OpenCV) sang RGB (YOLO)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Sử dụng mô hình để nhận diện vật thể
results = model(img_rgb)

# Kết quả trả về là danh sách, bạn cần lấy đối tượng đầu tiên
result = results[0]

# Hiển thị kết quả
result.show()  # Hiển thị hình ảnh với bounding boxes