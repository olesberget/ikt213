import cv2
import numpy as np
import img2pdf



def harris_corner_detection(image_path, save_path='harris.png'):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    image[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imwrite(save_path, image)



def sift_image_alignment(image_to_align_path, reference_image_path):
    image_to_align = cv2.imread(image_to_align_path)
    reference_image = cv2.imread(reference_image_path)

    gray_image_to_align = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    gray_reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    s = cv2.SIFT_create()
    kp1, d1 = s.detectAndCompute(gray_reference_image, None)
    kp2, d2 = s.detectAndCompute(gray_image_to_align, None)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = matcher.knnMatch(d1, d2, k=2)
    g_m = [m for m, n in matches if m.distance < 0.7 * n.distance]
    p1 = np.float32([kp1[m.queryIdx].pt for m in g_m]).reshape(-1, 1, 2)
    p2 = np.float32([kp2[m.trainIdx].pt for m in g_m]).reshape(-1, 1, 2)
    h, _ = cv2.findHomography(p2, p1, cv2.RANSAC)

    height, width, _ = reference_image.shape

    aligned_image = cv2.warpPerspective(image_to_align, h, (width, height))
    matches_img = cv2.drawMatches(reference_image, kp1, image_to_align, kp2, g_m, None)

    cv2.imwrite('aligned.png', aligned_image)
    cv2.imwrite('matches.png', matches_img)



def create_pdf(output_pdf='output.pdf', image_files=['harris.png', 'aligned.png', 'matches.png']):
    with open(output_pdf, "wb") as f:
        f.write(img2pdf.convert(image_files))



if __name__ == '__main__':
    reference_image = 'C:/Users/mini-/ikt213g24h/assignments/resources/reference_img.png'
    image_to_align = 'C:/Users/mini-/ikt213g24h/assignments/resources/align_this.jpg'

    harris_corner_detection(reference_image)
    sift_image_alignment(image_to_align, reference_image)

    create_pdf()