# Import các thư viện cần thiết
from inference_sdk import InferenceHTTPClient
import cv2

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="JPwevhyry3bT2iUBGeql"
)
# === Đường dẫn ảnh và model_id ===
image_file = r"image.png"
model_id = "lastdata-fmxrb/2"

# Chạy inference
result = CLIENT.infer(image_file, model_id=model_id)

# === Tải ảnh gốc ===
image = cv2.imread(image_file)

# === Duyệt qua các predictions và vẽ bounding boxes ===
for prediction in result['predictions']:
    # Lấy thông tin của mỗi bounding box
    x_center = prediction['x']
    y_center = prediction['y']
    width = prediction['width']
    height = prediction['height']
    confidence = prediction['confidence']
    class_name = prediction['class']

    # Tính toán tọa độ góc trên trái và dưới phải
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)

    # Vẽ bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Gắn nhãn class và confidence
    label = f"{class_name}: {confidence:.2f}"
    cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# === Hiển thị hoặc lưu ảnh đã annotate ===
output_file = "annotated_image.jpg"
cv2.imwrite(output_file, image)
print(f"Annotated image saved to {output_file}")

# Hiển thị ảnh (tuỳ chọn)
cv2.imshow("Annotated Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
