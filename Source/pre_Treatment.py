import os
import pandas as pd

# Hàm xử lý dữ liệu và lưu nhãn
def firstStep(file_pathIn, output_dirIn):
    file_path = file_pathIn
    data = pd.read_csv(file_path, header=None)
    img_width = 1024
    img_height = 540
    output_dir = output_dirIn
    os.makedirs(output_dir, exist_ok=True)
    for index, row in data.iterrows():
        frame_id = int(row[0])
        class_id = 0  
        x1, y1, w, h = row[2:6]  
        x_center = (x1 + w / 2) / img_width
        y_center = (y1 + h / 2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
        label_file_path = os.path.join(output_dir, f"img{frame_id:06d}.txt")
        with open(label_file_path, "a") as label_file:
            label_file.write(label_line)

    print(f"Labels saved in: {output_dir}")

# Hàm lấy danh sách các tệp trong thư mục
def list_files_in_folder(path):
    try:
        items = os.listdir(path)
        files = [item for item in items if os.path.isfile(os.path.join(path, item))]
        return files
    except FileNotFoundError:
        print("Thư mục không tồn tại.")
        return []

def process_files(input_folder, output_folder):
    files = list_files_in_folder(input_folder)

    for file_name in files:
        file_path = os.path.join(input_folder, file_name)
        subfolder_name = file_name[:5]
        subfolder_path = os.path.join(output_folder, subfolder_name)

        firstStep(file_path, subfolder_path)


# input_folder = r"D:\AvideodataCV\label"  
# output_folder = r"D:\AvideodataCV\Labels"  
# process_files(input_folder, output_folder)


import os
import shutil

def move_paired_files(path_txt, path_jpg, output_dir_label, output_dir_img):
    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(output_dir_label, exist_ok=True)
    os.makedirs(output_dir_img, exist_ok=True)
    
    # Bộ đếm để tạo tên tệp mới
    counter = 1

    # Lấy danh sách các thư mục con trong cả hai đường dẫn
    subfolders_txt = [f for f in os.listdir(path_txt) if os.path.isdir(os.path.join(path_txt, f))]
    subfolders_jpg = [f for f in os.listdir(path_jpg) if os.path.isdir(os.path.join(path_jpg, f))]
    
    # Lặp qua các thư mục trong A (path_txt)
    for folder in subfolders_txt:
        if folder in subfolders_jpg:  # Chỉ xử lý nếu thư mục tồn tại ở cả hai
            txt_folder = os.path.join(path_txt, folder)
            jpg_folder = os.path.join(path_jpg, folder)

            # Lấy danh sách tệp txt và jpg
            txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
            jpg_files = [f for f in os.listdir(jpg_folder) if f.endswith('.jpg')]

            # Tìm các tệp có tên giống nhau
            paired_files = set(f[:-4] for f in txt_files).intersection(f[:-4] for f in jpg_files)

            # Di chuyển từng cặp vào thư mục đích
            for pair in paired_files:
                txt_path = os.path.join(txt_folder, f"{pair}.txt")
                jpg_path = os.path.join(jpg_folder, f"{pair}.jpg")
                
                # Tên mới cho tệp
                new_name = f"{counter:06d}"
                
                # Di chuyển và đổi tên các tệp
                shutil.move(txt_path, os.path.join(output_dir_label, f"{new_name}.txt"))
                shutil.move(jpg_path, os.path.join(output_dir_img, f"{new_name}.jpg"))

                print(f"Moved: {new_name}.txt, {new_name}.jpg")
                counter += 1

# # Ví dụ sử dụng
# path_txt = r"D:\AvideodataCV\Labels"  # Đường dẫn chứa các thư mục con với tệp txt
# path_jpg = r"D:\AvideodataCV\UAV-benchmark-M\UAV-benchmark-M"  # Đường dẫn chứa các thư mục con với tệp jpg

# output_dirL = r"D:\AvideodataCV\data\label"  # Thư mục đích để lưu các tệp nhãn
# output_dirI = r"D:\AvideodataCV\data\img"  # Thư mục đích để lưu các tệp hình ảnh
# move_paired_files(path_txt, path_jpg, output_dirL, output_dirI)



import os
import shutil
import random

def move():

    folder_A = "D:\AvideodataCV\dataset\label"  # Thư mục chứa các file txt
    folder_B = "D:\AvideodataCV\dataset\img"  # Thư mục chứa các file jpg
    folder_C = "D:\AvideodataCV\data\label"  # Thư mục đích cho file txt
    folder_D = "D:\AvideodataCV\data\img"  # Thư mục đích cho file jpg

    # Tạo thư mục C và D nếu chưa tồn tại
    os.makedirs(folder_C, exist_ok=True)
    os.makedirs(folder_D, exist_ok=True)

    # Lấy danh sách các file trong thư mục A và B
    txt_files = {os.path.splitext(file)[0]: file for file in os.listdir(folder_A) if file.endswith('.txt')}
    jpg_files = {os.path.splitext(file)[0]: file for file in os.listdir(folder_B) if file.endswith('.jpg')}

    # Tìm các file có cùng tên (cặp)
    common_files = list(set(txt_files.keys()) & set(jpg_files.keys()))

    # Chọn ngẫu nhiên 2000 cặp
    selected_files = random.sample(common_files, min(2000, len(common_files)))

    # Di chuyển các file được chọn sang thư mục C và D
    for file_name in selected_files:
        # Di chuyển file txt
        src_txt = os.path.join(folder_A, txt_files[file_name])
        dest_txt = os.path.join(folder_C, txt_files[file_name])
        shutil.move(src_txt, dest_txt)
        
        # Di chuyển file jpg
        src_jpg = os.path.join(folder_B, jpg_files[file_name])
        dest_jpg = os.path.join(folder_D, jpg_files[file_name])
        shutil.move(src_jpg, dest_jpg)

    print(f"Đã chuyển thành công {len(selected_files)} cặp tệp vào thư mục C và D.")


import cv2

def draw_bounding_box(image_path, label, x, y, width, height, output_path="output.jpg"):

    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("")
    
    # Tính toán tọa độ góc trên trái và dưới phải từ x_center, y_center, width, height
    x_min = int(x - width / 2)
    y_min = int(y - height / 2)
    x_max = int(x + width / 2)
    y_max = int(y + height / 2)
    
    # Vẽ bounding box (màu xanh lá cây)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # Gắn nhãn (class name) trên bounding box
    label_text = f"{label}"
    cv2.putText(image, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Lưu ảnh đã annotate
    cv2.imwrite(output_path, image)
    print(f"Ảnh đã được lưu tại: {output_path}")
    
    # Hiển thị ảnh kết quả (tuỳ chọn)
    cv2.imshow("Annotated Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ví dụ sử dụng
draw_bounding_box(
    image_path="001235.jpg",  # Đường dẫn tới ảnh gốc
    label="car",            # Tên nhãn
    x=380,                  # Tọa độ trung tâm x
    y=330,                  # Tọa độ trung tâm y
    width=50,              # Chiều rộng bounding box
    height=120               # Chiều cao bounding box
)
