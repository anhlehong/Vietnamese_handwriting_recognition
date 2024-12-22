import cv2
import numpy as np
from app.utils.image_processing import rotation, crop_whitespace, change_background_to_white
import os

def process_and_invert_image(image_path, output_folder):

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")

    # Chuyển đổi ảnh sang grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Áp dụng Gaussian Blur để làm mượt ảnh
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Áp dụng adaptive threshold để làm nổi bật chữ trắng trên nền đen
    thresholded_img = cv2.adaptiveThreshold(
        blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4
    )

    # Biến đổi toàn bộ pixel trắng thành đen và pixel đen thành trắng
    inverted_img = cv2.bitwise_not(thresholded_img)

    # Resize ảnh về chiều rộng mong muốn (new_width)
    new_width = 1795
    height, width = inverted_img.shape[:2]  # Lấy kích thước gốc
    aspect_ratio = new_width / width  # Tỷ lệ thay đổi chiều rộng
    new_height = int(height * aspect_ratio)  # Tính chiều cao mới
    resized_img = cv2.resize(
        inverted_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Tạo đường dẫn lưu ảnh
    output_path = os.path.join(
        output_folder, f"processed_{os.path.basename(image_path)}"
    )

    # Lưu ảnh kết quả
    cv2.imwrite(output_path, resized_img)

    return output_path


def extract_lines(image, output_folder, output_width=1854, output_height=103, background_color=(255, 255, 255)):
    """
    Tách văn bản thành các dòng và lưu thành từng ảnh.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 200, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 10))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    processed_images = []
    lines = []
    index = 1
    for i, ctr in enumerate(contours):
        x, y, w, h = cv2.boundingRect(ctr)
        if h < 50:
            continue

        cropped = image[y:y+h, x:x+w]
        cropped = crop_whitespace(cropped)
        rotated = rotation(cropped)
        final_image = change_background_to_white(rotated)

        h, w, _ = final_image.shape
        if w < output_width:
            new_image = np.full((output_height, output_width, 3),
                                background_color, dtype=np.uint8)
            if h > output_height:
                # Resize nếu chiều cao lớn hơn chiều cao mong muốn
                resized_image = cv2.resize(
                    final_image, (w, output_height), interpolation=cv2.INTER_AREA)
                new_image[:output_height, :w] = resized_image
            else:
                new_image[:h, :w] = final_image
        else:
            new_image = cv2.resize(
                final_image, (output_width, output_height), interpolation=cv2.INTER_AREA)

        output_path = f"{output_folder}/line_{index}.png"
        cv2.imwrite(output_path, new_image)
        processed_images.append(output_path)
        lines.append(new_image)
        index = index + 1

    return processed_images, lines

