import cv2
import numpy as np
import os
from app.utils.image_processing import rotation, change_background_to_white, crop_whitespace, shadow_removal

def crop_line(image, output_folder, image_path):
    
    light = shadow_removal(image)

    output_path = os.path.join(
        output_folder, f"processed_{os.path.basename(image_path)}"
    )
    cv2.imwrite(output_path, light)

    # gray = cv2.cvtColor(light, cv2.COLOR_BGR2GRAY)
    
    gray = rotation(light)

    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 17))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Tìm contour của các dòng chữ
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sắp xếp các contour từ trên xuống dưới
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    # Kích thước đầu ra
    output_width = 1854
    output_height = 103
    background_color = (255, 255, 255)  # Màu nền (trắng)

    id_image = 3919

    light = cv2.cvtColor(light, cv2.COLOR_GRAY2BGR)

    texts = []

    for i, ctr in enumerate(contours):
        x, y, w, h = cv2.boundingRect(ctr)

        if h < 40: 
            continue

        # Cắt dòng chữ từ ảnh gốc
        cropped = light[y:y+h, x:x+w]

        # Cắt khoảng trắng trên, dưới và bên trái
        cropped = crop_whitespace(cropped)

        # Xoay ảnh để chỉnh lại góc nghiêng
        rotated_image = rotation(cropped)

        # Sau khi xoay, tiếp tục cắt khoảng trắng trên, dưới và bên trái
        cut_image = change_background_to_white(rotated_image, threshold=135)
        final_image = cut_image
        

        # Xử lý chiều rộng và chiều cao cố định
        h, w, _ = final_image.shape
        if w < output_width:
            # Tạo ảnh với chiều rộng cố định
            if h < 102:  # Thêm padding để đảm bảo chiều cao >= 102
                padding_top = (102 - h) // 2
                padding_bottom = 102 - h - padding_top
                new_image = np.full((102, output_width, 3), background_color, dtype=np.uint8)
                new_image[padding_top:padding_top + h, :w] = final_image  # Chèn chữ vào giữa
            elif h > 5:  # Resize chiều cao về 118
                scale = 105 / h
                new_width = int(w * scale)
                resized_image = cv2.resize(final_image, (new_width, 105), interpolation=cv2.INTER_LINEAR)
                new_image = np.full((105, output_width, 3), background_color, dtype=np.uint8)
                new_image[:, :new_width] = resized_image  # Chèn ảnh đã resize vào trái
            else:  # Nếu chiều cao trong khoảng 102 <= h <= 118
                new_image = np.full((h, output_width, 3), background_color, dtype=np.uint8)
                new_image[:h, :w] = final_image
        else:
            new_image = final_image

        # Hiển thị ảnh đã xử lý
        # plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
        # plt.title(f"Dòng {i+1} sau xử lý kích thước cố định")
        # plt.axis("off")
        # plt.show()
        texts.append(new_image)
    
    return texts, output_path

def preprocess_line(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 17))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    output_width = 1854
    output_height = 103
    background_color = (255, 255, 255)  


    for i, ctr in enumerate(contours):
        x, y, w, h = cv2.boundingRect(ctr)

        if h < 40: 
            continue

        # Cắt dòng chữ từ ảnh gốc
        cropped = image[y:y+h, x:x+w]

        # Cắt khoảng trắng trên, dưới và bên trái
        cropped = crop_whitespace(cropped)

        # Xoay ảnh để chỉnh lại góc nghiêng
        rotated_image = rotation(cropped)

        # Sau khi xoay, tiếp tục cắt khoảng trắng trên, dưới và bên trái
        cut_image = change_background_to_white(rotated_image, threshold=250)
        final_image = cut_image

        # Xử lý chiều rộng và chiều cao cố định
        h, w, _ = final_image.shape
        if w < output_width:
            # Tạo ảnh với chiều rộng cố định
            if h < 102:  # Thêm padding để đảm bảo chiều cao >= 102
                padding_top = (102 - h) // 2
                padding_bottom = 102 - h - padding_top
                new_image = np.full((102, output_width, 3), background_color, dtype=np.uint8)
                new_image[padding_top:padding_top + h, :w] = final_image  # Chèn chữ vào giữa
            elif h > 5:  # Resize chiều cao về 118
                scale = 105 / h
                new_width = int(w * scale)
                resized_image = cv2.resize(final_image, (new_width, 105), interpolation=cv2.INTER_LINEAR)
                new_image = np.full((105, output_width, 3), background_color, dtype=np.uint8)
                new_image[:, :new_width] = resized_image  # Chèn ảnh đã resize vào trái
            else:  # Nếu chiều cao trong khoảng 102 <= h <= 118
                new_image = np.full((h, output_width, 3), background_color, dtype=np.uint8)
                new_image[:h, :w] = final_image
        else:

            new_image = final_image

    return new_image