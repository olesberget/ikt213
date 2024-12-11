import cv2
import numpy as np

def padding(image, border_width=100):
    padded = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)
    return padded

def crop(image, x_0=80, x_1=130, y_0=80, y_1=130):
    cropped = image[y_0:y_1, x_0:x_1]
    return cropped

def resize(image, width=200, height=200):
    resized = cv2.resize(image, (width, height))
    return resized

def copy(image, emptyPictureArray):
    height, width, channels = image.shape
    for i in range(height):
        for j in range(width):
            emptyPictureArray[i, j] = image[i, j]
    return emptyPictureArray

def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv


def hue_shifted(image, emptyPictureArray, hue=50):
    shifted = image.astype(np.int16)
    shifted[:, :, 0] = (shifted[:, :, 0] + hue)
    shifted[:, :, 1] = (shifted[:, :, 1] + hue)
    shifted[:, :, 2] = (shifted[:, :, 2] + hue)
    shifted_image = np.clip(shifted, 0, 255).astype(np.uint8)
    np.copyto(emptyPictureArray, shifted_image)
    return emptyPictureArray


def smoothing(image):
    smoothed = cv2.GaussianBlur(image,(15,15),cv2.BORDER_DEFAULT)
    return smoothed

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
    else:
        raise ValueError("Invalid rotation angle!")
    return rotated


if __name__ == "__main__":

    image_path = "../../resources/lena.png"
    image = cv2.imread(image_path)

    padded = padding(image)
    cropped = crop(image)
    resized = resize(image)

    emptyPictureArray = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    copied = copy(image, emptyPictureArray)

    gray = grayscale(image)
    hsv_img = hsv(image)
    hue_shifted_img = hue_shifted(image, emptyPictureArray)
    smoothed = smoothing(image)
    rotated_90 = rotation(image, 90)
    rotated_180 = rotation(image, 180)