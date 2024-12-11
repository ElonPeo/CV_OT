from inference_sdk import InferenceHTTPClient
import cv2
import os

# === Khởi tạo Roboflow Client ===
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="JPwevhyry3bT2iUBGeql"
)

# === Đường dẫn video đầu vào và đầu ra ===
video_file = r"b.mp4"  # Đường dẫn video gốc
output_video = "annotated_video.mp4"  # Video đầu ra sau khi annotate
frame_output_dir = "frames"  # Thư mục lưu các khung hình tạm thời

# === Đảm bảo thư mục tạm thời tồn tại ===
os.makedirs(frame_output_dir, exist_ok=True)

# === Tách các khung hình từ video ===
cap = cv2.VideoCapture(video_file)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Video writer cho video đầu ra
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

frame_index = 0
max_frames_to_infer = 200  # Giới hạn số khung hình gửi lên
processed_frames = 0

while cap.isOpened() and processed_frames < max_frames_to_infer:
    ret, frame = cap.read()
    if not ret:
        break

    # Lưu khung hình tạm thời
    frame_path = os.path.join(frame_output_dir, f"frame_{frame_index}.jpg")
    cv2.imwrite(frame_path, frame)

    # Gửi frame đến Roboflow để lấy kết quả
    result = CLIENT.infer(frame_path, model_id="lastdata-fmxrb/2")

    # Duyệt qua các predictions và vẽ bounding boxes
    for prediction in result['predictions']:
        x_center = prediction['x']
        y_center = prediction['y']
        width = prediction['width']
        height = prediction['height']
        confidence = prediction['confidence']
        class_name = prediction['class']

        # Tính toán tọa độ góc bounding box
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        # Vẽ bounding box lên frame
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Gắn nhãn class và confidence
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Thêm frame đã annotate vào video đầu ra
    out.write(frame)
    processed_frames += 1
    frame_index += 1
    print(f"Processed frame {processed_frames}/{max_frames_to_infer}")

# === Giải phóng tài nguyên ===
cap.release()
out.release()
cv2.destroyAllWindows()

# Xoá các khung hình tạm thời nếu cần
for f in os.listdir(frame_output_dir):
    os.remove(os.path.join(frame_output_dir, f))
os.rmdir(frame_output_dir)

print(f"Annotated video saved to {output_video}")
