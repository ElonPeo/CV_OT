import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import time

class ObjectDetection():
    def __init__(self, video_path, output_path):
        self.model = self.loadModel()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.cap = cv2.VideoCapture(video_path)  
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        fps = self.cap.get(cv2.CAP_PROP_FPS)  # Lấy FPS của video gốc
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.7)
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.7)
        self.out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # Tạo VideoWriter

        self.tracker = DeepSort(
            max_age=3,
            n_init=3,
            nms_max_overlap=0.3,
            max_cosine_distance=0.3,  
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
        self.start_time = time.time()

    def loadModel(self):
        model = YOLO(r"checkpoint\My_model\RUN_20241118_190549_822751\models\best.pt") 
        model.to('cuda')  
        return model

    def predict(self, img):
        results = self.model(img, stream=True, verbose=False)
        return results

    def plotBoxes(self, results, img):
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if conf > 0.4:  
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))
        return detections, img

    def trackDetect(self, detections, img):
        # Cập nhật các đối tượng được theo dõi
        tracks = self.tracker.update_tracks(detections, frame=img)
        for track in tracks:
            if not track.is_confirmed():
                continue
            # Lấy thông tin đối tượng
            track_id = track.track_id
            ltrb = track.to_ltrb()  # (left, top, right, bottom)
            x1, y1, x2, y2 = map(int, ltrb)  # Chuyển đổi sang int

            # Vẽ bounding box (khung viền)
            color = (255, 0, 0)  # Màu viền (xanh dương)
            thickness = 2  # Độ dày viền
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # (Tùy chọn) Hiển thị ID đối tượng
            text = f"ID: {track_id}"
            font_scale = 0.5
            font_thickness = 1
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

        return img


    def __call__(self):
        # Lặp qua các khung hình của video
        while self.cap.isOpened():
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 100:  # Dừng sau 100 giây nếu cần
                break

            ret, img = self.cap.read()  # Đọc khung hình từ video
            if not ret:
                break

            img = cv2.resize(img, (0, 0), fx=0.7, fy=0.7)  # Giảm kích thước khung hình

            results = self.predict(img)
            for result in results:
                detections, frames = self.plotBoxes([result], img)
                detect_frame = self.trackDetect(detections, frames)

                # Ghi khung hình đã xử lý vào tệp video đầu ra
                self.out.write(detect_frame)

                # Hiển thị khung hình
                cv2.imshow('Tracking', detect_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Giải phóng tài nguyên
        self.cap.release()
        self.out.release()  # Đảm bảo lưu video đầy đủ
        cv2.destroyAllWindows()

# Đường dẫn tới video cần xử lý
video_path = r'b.mp4'
# Đường dẫn tới video đầu ra
output_path = r'outvideo'

detector = ObjectDetection(video_path, output_path)
detector()
