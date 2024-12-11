import cv2
import numpy as np

def sobel_edge_detection(image):
    blurred = cv2.GaussianBlur(image, (3, 3), sigmaX=0)
    sobel = cv2.Sobel(blurred, cv2.CV_64F, 1, 1, ksize=1)
    return sobel


def canny_edge_detection(image, threshold_1=50, threshold_2=50):
    blurred = cv2.GaussianBlur(image, (3, 3), sigmaX=0)
    canny = cv2.Canny(blurred, threshold_1, threshold_2)
    return canny


def template_match(image, template):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    temp_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(img_gray, temp_gray, cv2.TM_CCOEFF_NORMED)

    threshold = 0.9
    loc = np.where(result >= threshold)

    h, w = temp_gray.shape
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    return image


def resize(image, scale_factor: int, up_or_down: str):
    if up_or_down == "up":
        resized = cv2.pyrUp(image, dstsize=(image.shape[1] * scale_factor, image.shape[0] * scale_factor))
    elif up_or_down == "down":
        resized = cv2.pyrDown(image, dstsize=(image.shape[1] // scale_factor, image.shape[0] // scale_factor))
    return resized


if __name__ == "__main__":
    image = cv2.imread('../../resources/lambo.png')
    template = cv2.imread('../../resources/shapes.png')

    sobel = sobel_edge_detection(image)

    canny = canny_edge_detection(image)

    matched = template_match(image, template)

    resized = resize(image, 2, "up")

    cv2.waitKey(0)
    cv2.destroyAllWindows()