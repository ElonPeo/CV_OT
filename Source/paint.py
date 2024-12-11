import torch
from PIL import Image
from torchvision import transforms
from super_gradients.training import models

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CLASSES = ['0']  # Giả sử mô hình nhận diện 1 lớp


    # Tải mô hình
    model = models.get(
        model_name='yolo_nas_s',
        num_classes=len(CLASSES),
        checkpoint_path=r"checkpoint\My_model\RUN_20241117_235257_452431\ckpt_best.pth"
    ).to(DEVICE)
    model.eval()  # Đặt mô hình vào chế độ đánh giá

    # Tiền xử lý hình ảnh
    image_path = r'Data\Images\img000001.jpg'  # Đường dẫn đến hình ảnh
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)  # Thêm batch dimension

    # Dự đoán
    with torch.no_grad():
        predictions = model(input_tensor)

    # Kiểm tra predictions
    print(predictions)



if __name__ == '__main__':
    main()
