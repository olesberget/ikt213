import os
import cv2

def print_image_information(image):
    height, width, channels = image.shape
    size = image.size
    data_type = image.dtype

    print(f"height: {height}")
    print(f"width: {width}")
    print(f"channels: {channels}")
    print(f"size: {size}")
    print(f"data type: {data_type}")

def save_camera_info(camera):
    fps = camera.get(cv2.CAP_PROP_FPS)
    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

    info = f"fps: {int(fps)}\nwidth: {int(width)}\nheight: {int(height)}"

    file_path = os.path.join(os.path.dirname(__file__), "camera_outputs.txt")
    with open(file_path, 'w') as file:
        file.write(info)

def main():
    image_path = "C:/Users/mini-/ikt213g24h/assignments/resources/lena.png"
    image = cv2.imread(image_path)
    camera = cv2.VideoCapture(0)

    print_image_information(image)
    save_camera_info(camera)

if __name__ == "__main__":
    main()