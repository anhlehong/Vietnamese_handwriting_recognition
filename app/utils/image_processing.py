import cv2
import numpy as np
import os

def preprocess_image(image_input, resize_max_width=2167):
    # Đọc ảnh
    # img = cv2.imread(image_input)
    if isinstance(image_input, str) and os.path.isfile(image_input):
        # Đọc ảnh từ đường dẫn
        img = cv2.imread(image_input)
    elif isinstance(image_input, np.ndarray):
        # Nếu input là mảng ảnh (NumPy array)
        img = image_input
    else:
        raise ValueError("Input must be a valid file path or a NumPy image array.")
    
    # Chuyển ảnh từ RGB sang grayscale (1 kênh)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Không thay đổi kích thước ảnh theo chiều cao và chiều rộng trong dataset này
    img = cv2.resize(img, (int(118 / img.shape[0] * img.shape[1]), 118))

    # Điều chỉnh kích thước ảnh nếu cần
    if img.shape[1] > resize_max_width:
        resize_max_width = img.shape[1]

    # Thêm padding nếu ảnh nhỏ hơn chiều rộng mong muốn
    img = np.pad(img, ((0, 0), (0, resize_max_width - img.shape[1])), 'median')

    # Làm mờ ảnh
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Áp dụng threshold thích ứng
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    # Thêm chiều kênh (1 kênh)
    img = np.expand_dims(img, axis=2)

    # Chuẩn hóa ảnh
    img = img / 255.0
    
    # Thêm chiều batch (1, 118, 2167, 1)
    img = np.expand_dims(img, axis=0)
    
    return img


def rotation(img):
    """
    Xoay ảnh dựa trên góc nghiêng.
    """
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 60)

    if lines is not None:
        angles = [
            (theta - np.pi / 2) * (180 / np.pi)
            for rho, theta in lines[:, 0]
            if -10 <= (theta - np.pi / 2) * (180 / np.pi) <= 10
        ]
        if angles:
            average_angle = np.mean(angles)
            (h, w) = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D(
                (w // 2, h // 2), average_angle, 1.0)
            rotated = cv2.warpAffine(
                img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
            return rotated
    return img


def crop_whitespace(image, background_color=(255, 255, 255), left_padding=10):
    """
    Cắt khoảng trắng khỏi ảnh.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray < 250
    coords = cv2.findNonZero(mask.astype(np.uint8))
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        x = max(0, x - left_padding)
        cropped = image[y:y+h, x:x+w+10]
        return cropped
    return image


def change_background_to_white(image, threshold=200):
    """
    Chuyển nền ảnh thành trắng.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    image[mask] = [255, 255, 255]
    return image

def add_padding(image, target_width, target_height, background_color):
    original_height, original_width = image.shape[:2]

    # Nếu chiều rộng lớn hơn 1854, resize về 1854 và tính chiều cao tương ứng
    if original_width > 1854:
        scaling_factor = 1854 / original_width
        new_width = 1854
        new_height = int(original_height * scaling_factor)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        original_height, original_width = image.shape[:2]

    # Tính toán padding
    padding_top = (target_height - original_height) // 2
    padding_bottom = target_height - original_height - padding_top
    padding_left = 0
    padding_right = target_width - original_width

    if padding_right < 0 or padding_bottom < 0 or padding_top < 0:
        raise ValueError("Target size must be larger than the original size.")

    # Thêm padding vào ảnh
    padded_image = cv2.copyMakeBorder(
        image,
        top=padding_top,
        bottom=padding_bottom,
        left=padding_left,
        right=padding_right,
        borderType=cv2.BORDER_CONSTANT,
        value=background_color
    )

    return padded_image

def shadow_removal(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Áp dụng Otsu Thresholding để tách nền và chữ
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Làm mờ nền bằng Gaussian Blur
    blurred = cv2.GaussianBlur(image, (21, 21), 0)

    # Trừ nền để giữ lại chi tiết
    normalized = cv2.divide(image, blurred, scale=255)

    # Làm sáng nền bằng cách tăng giá trị cường độ sáng
    bright_background = cv2.add(normalized, 20)

    # Chỉnh gamma để làm sáng nền
    gamma = 1.5
    gamma_correction = ((bright_background / 255.0) ** (1 / gamma)) * 255
    bright_corrected = gamma_correction.astype(np.uint8)

    # Chỉ áp dụng làm sáng lên nền bằng cách sử dụng mask
    result = cv2.bitwise_and(bright_corrected, bright_corrected, mask=~binary)
    result = cv2.bitwise_or(result, binary)

    threshold = 240  # Ngưỡng để xác định màu trắng
    binary_to_black = np.where(result < threshold, 0, 255).astype("uint8")

    return binary_to_black