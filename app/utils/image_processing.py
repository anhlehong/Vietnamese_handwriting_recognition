import cv2
import numpy as np


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

