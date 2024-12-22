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

