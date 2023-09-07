import cv2
import numpy as np
from matplotlib import pyplot as plt


def load_image(image_name):
    image = cv2.imread(image_name)
    return image


def resize_image(image):
    max_dim = max(image.shape)
    scale = 700 / max_dim
    image = cv2.resize(image, None, fx=scale, fy=scale)
    return image


def blur_image(image):
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)
    return image_blur_hsv


def make_filter(image_blur_hsv):
    # Filters
    color_lower_bound = np.array([30, 150, 130])
    color_upper_bound = np.array([45, 256, 256])

    mask_color = cv2.inRange(image_blur_hsv, color_lower_bound, color_upper_bound)

    brightness_lower_bound = np.array([0, 100, 80])
    brightness_upper_bound = np.array([180, 256, 240])

    mask_brightness = cv2.inRange(image_blur_hsv, brightness_lower_bound, brightness_upper_bound)
    # Combine masks
    mask = mask_color + mask_brightness

    # Segment
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    return mask_opened


def overlay_filter(mask, image):
    # Make mask RGB
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img


def find_biggest_contour(image):
    # Convert Image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a Mask with adaptive threshold
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #image1 = image.copy()
    #contours, __ = cv2.findContours(image1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #cv2.findContours(image1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda element: element[0])[1]
    return biggest_contour


def draw_rect_box(contour, image):
    rect = cv2.minAreaRect(contour)
    (x, y), (w, h), angle = rect
    # Display rectangle
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.polylines(image, [box], True, (255, 0, 0), 2)
    return image


def show(image):
    plt.figure(figsize=[10, 10])
    plt.imshow(image, interpolation='nearest')


def detect_image(image):

    image = resize_image(image)
    image = blur_image(image)
    #mask = make_filter(image)
    #image = overlay_filter(mask, image)
    contour = find_biggest_contour(image)
    image = draw_rect_box(contour , image)
    show(image)
    return image


img = load_image('img1/1.jpg')
after_img = detect_image(img)
cv2.imwrite('output4/1.jpg', after_img)