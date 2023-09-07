import cv2
import numpy as np


def load_image(image_name):
    image = cv2.imread(image_name)
    return image


def resize_image(image):
    max_dim = max(image.shape)
    scale = 800 / max_dim
    image = cv2.resize(image, None, fx=scale, fy=scale)
    return image


parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)


def detect_object():
    aruco_x, aruco_y, pixel_cm_ratio = detect_aruco()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ------- threshold-----------------
    contours = find_contours_by_threshold(gray)
    # --------Canny----------------------
    # contours = find_contours_by_canny(gray)

    objects_contours = find_object_contours(aruco_x, aruco_y, contours)

    measure_objects(objects_contours, pixel_cm_ratio)


def measure_objects(objects_contours, pixel_cm_ratio):
    for cnt in objects_contours:
        # Get rect
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect
        # Get Width and Height of the Objects by applying the Ratio pixel to cm
        object_width = w / pixel_cm_ratio
        object_height = h / pixel_cm_ratio
        # Display rectangle
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.polylines(img, [box], True, (255, 0, 0), 2)
        cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 200)),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y - 150)),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        f.write(str(round(object_width, 1)) + '\t \t')
        f.write(str(round(object_height, 1)) + '\n')


def find_object_contours(aruco_x, aruco_y, contours):
    objects_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2500:
            # Get rect
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect
            # print('x:' + str(round(x)))
            # print('y:' + str(round(y)))

            # remove aruco from detected contours
            if abs(round(x) - round(aruco_x)) > 10 or abs(round(y) - round(aruco_y)) > 10:
                objects_contours.append(cnt)

    print(len(objects_contours), "objects were found in this image.")
    return objects_contours


def find_contours_by_canny(gray):
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 10, 100)
    # define a (3, 3) structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # apply the dilation operation to the edged image
    dilate = cv2.dilate(edged, kernel, iterations=1)
    # find the contours in the dilated image
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def find_contours_by_threshold(gray):
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def detect_aruco():
    corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    int_corners = np.intp(corners)
    cv2.polylines(img, int_corners, True, (0, 255, 0), 5)
    aruco_contour = corners[0];
    aruco_perimeter = cv2.arcLength(corners[0], True)
    pixel_cm_ratio = aruco_perimeter / 18
    aruco_rect = cv2.minAreaRect(aruco_contour)
    (aruco_x, aruco_y), (aruco_w, aruco_h), aruco_angle = aruco_rect
    cv2.circle(img, (int(aruco_x), int(aruco_y)), 5, (0, 255, 0), -1)
    # print('aruco_x:' + str(round(aruco_x)))
    # print('aruco_y:' + str(round(aruco_y)))
    return aruco_x, aruco_y, pixel_cm_ratio


# ---------  main ----------------------------
with open('output5/dimension.txt', 'w') as f:
    for xi in range(1, 29):
        name = str(xi) + ".jpg"
        f.write('\n --------------------------- \n' + name + '\n --------------------------- \n')

        img = load_image("img1/" + name)
        img = resize_image(img)

        detect_object()

        cv2.imwrite("output5/" + name, img)
        cv2.waitKey(0)
