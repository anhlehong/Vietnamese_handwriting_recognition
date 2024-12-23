import cv2
import numpy as np
from app.utils.image_processing import rotation, crop_whitespace, change_background_to_white, add_padding
from app.services.line_extractrion import preprocess_line, crop_line
import os

def rotation_word(img):
    """
    Điều chỉnh góc nghiêng của ảnh dựa trên các đường ngang.
    :param img: Ảnh đầu vào (NumPy array)
    :return: Ảnh đã được xoay
    """
    blurred = cv2.GaussianBlur(img, (25, 3), 0)
    _, binary = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    edges = cv2.Canny(dilated, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 60)

    if lines is not None:
        angles = [
            (theta - np.pi / 2) * (180 / np.pi)  # Tính góc
            for rho, theta in lines[:, 0]
            if -10 <= (theta - np.pi / 2) * (180 / np.pi) <= 10
        ]

        if angles: 
            average_angle = np.mean(angles)  # Tính góc trung bình

            # Tạo ma trận xoay để chỉnh ảnh
            (h, w) = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), average_angle, 1.0)
            rotated = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
            return rotated

    return img

def preprocess_word(image, inp_height = 100, ker_width = 50, ker_height = 30):
    image = preprocess_line(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ker_width, ker_height))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sắp xếp các contour từ trên xuống dưới
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bounding_boxes.append((x, y, w, h))

    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])


    words = []

    for i, (x, y, w, h) in enumerate(bounding_boxes):

        if h < 10: 
            continue

        cropped = image[y:y+h, x:x+w]


        rotated_image = rotation_word(cropped)

        words.append(rotated_image)


    max_height = max([word.shape[0] for word in words])

    padded_words = []

    for word in words:
        word_height = word.shape[0]
        padding_top = (max_height - word_height) // 2
        padding_bottom = max_height - word_height - padding_top
        
        padded_word = cv2.copyMakeBorder(
            word, 
            padding_top,        
            padding_bottom,
            20,                 
            0,                 
            cv2.BORDER_CONSTANT, 
            value=(255, 255, 255)
        )
        padded_words.append(padded_word)

    line_image = np.hstack(padded_words)
    line_image = crop_whitespace(line_image)

    new_height = inp_height 

    (h, w) = line_image.shape[:2]
    aspect_ratio = w / h
    new_width = int(new_height * aspect_ratio)

    # Resize ảnh
    resized_image = cv2.resize(line_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    result_image = add_padding(
        resized_image,
        target_width=1854,
        target_height=102,
        background_color=[255, 255, 255], 
    )

    return result_image

def text_output(image_path, output_folder, processed_folder): 
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
    lines, output_process_path = crop_line(image, processed_folder, image_path)

    processed_images = []
    texts = []
    index = 1
    for line in lines:
        new_image = preprocess_word(line, 100, 15, 25)
        output_path = f"{output_folder}/line_{index}.png"
        # print(output_path)
        cv2.imwrite(output_path, new_image)
        processed_images.append(output_path)
        texts.append(new_image)
        index = index + 1

    return output_process_path, processed_images, texts

# def process_and_invert_image(image_path, output_folder):

#     img = cv2.imread(image_path)
#     if img is None:
#         raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")

#     # Chuyển đổi ảnh sang grayscale
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Áp dụng Gaussian Blur để làm mượt ảnh
#     blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

#     # Áp dụng adaptive threshold để làm nổi bật chữ trắng trên nền đen
#     thresholded_img = cv2.adaptiveThreshold(
#         blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4
#     )

#     # Biến đổi toàn bộ pixel trắng thành đen và pixel đen thành trắng
#     inverted_img = cv2.bitwise_not(thresholded_img)

#     # Resize ảnh về chiều rộng mong muốn (new_width)
#     new_width = 1795
#     height, width = inverted_img.shape[:2]  # Lấy kích thước gốc
#     aspect_ratio = new_width / width  # Tỷ lệ thay đổi chiều rộng
#     new_height = int(height * aspect_ratio)  # Tính chiều cao mới
#     resized_img = cv2.resize(
#         inverted_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

#     # Tạo đường dẫn lưu ảnh
#     output_path = os.path.join(
#         output_folder, f"processed_{os.path.basename(image_path)}"
#     )

#     # Lưu ảnh kết quả
#     cv2.imwrite(output_path, resized_img)

#     return output_path


# def extract_lines(image, output_folder, output_width=1854, output_height=103, background_color=(255, 255, 255)):
#     """
#     Tách văn bản thành các dòng và lưu thành từng ảnh.
#     """
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 50, 200, cv2.THRESH_BINARY_INV)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 10))
#     dilated = cv2.dilate(binary, kernel, iterations=1)
#     contours, _ = cv2.findContours(
#         dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

#     processed_images = []
#     lines = []
#     index = 1
#     for i, ctr in enumerate(contours):
#         x, y, w, h = cv2.boundingRect(ctr)
#         if h < 50:
#             continue

#         cropped = image[y:y+h, x:x+w]
#         cropped = crop_whitespace(cropped)
#         rotated = rotation(cropped)
#         final_image = change_background_to_white(rotated)

#         h, w, _ = final_image.shape
#         if w < output_width:
#             new_image = np.full((output_height, output_width, 3),
#                                 background_color, dtype=np.uint8)
#             if h > output_height:
#                 # Resize nếu chiều cao lớn hơn chiều cao mong muốn
#                 resized_image = cv2.resize(
#                     final_image, (w, output_height), interpolation=cv2.INTER_AREA)
#                 new_image[:output_height, :w] = resized_image
#             else:
#                 new_image[:h, :w] = final_image
#         else:
#             new_image = cv2.resize(
#                 final_image, (output_width, output_height), interpolation=cv2.INTER_AREA)

#         output_path = f"{output_folder}/line_{index}.png"
#         cv2.imwrite(output_path, new_image)
#         processed_images.append(output_path)
#         lines.append(new_image)
#         index = index + 1

#     return processed_images, lines

