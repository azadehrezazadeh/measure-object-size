import cv2
import imutils
from object_detector import *
import numpy as np

# Load Aruco detector
parameters = cv2.aruco.DetectorParameters();
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50);

# Load Object Detector
detector = HomogeneousBgDetector()

with open('output2/dimension.txt', 'w') as f:

    for xi in range(1, 23):
        name = str(xi) + ".jpg";
        f.write('\n --------------------------- \n'+ name+'\n --------------------------- \n');

        # Load Image
        img = cv2.imread("img1/" + name)
        h, w, c = img.shape

        if h >= w:
            # img = imutils.rotate_bound(img,90)
            img = imutils.resize(img, height=900, width=600)
        else:
            img = imutils.resize(img, height=600, width=900)

        # Get Aruco marker
        corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

        # Draw polygon around the marker
        int_corners = np.int0(corners)
        cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

        # Aruco Perimeter
        aruco_perimeter = cv2.arcLength(corners[0], True)

        # Pixel to cm ratio
        pixel_cm_ratio = aruco_perimeter / 18

        contours = detector.detect_objects(img)

        # Draw objects boundaries
        for cnt in contours:
            # Get rect
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect

            # Get Width and Height of the Objects by applying the Ratio pixel to cm
            object_width = w / pixel_cm_ratio
            object_height = h / pixel_cm_ratio

            # Display rectangle
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.polylines(img, [box], True, (255, 0, 0), 2)
            cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 100)),
                        cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 255, 0), 2)
            cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y - 80)),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            f.write(str(round(object_width, 1)) + '\t \t')
            f.write(str(round(object_height, 1)) + '\n')

        cv2.imwrite("output2/" + name, img)
        cv2.waitKey(0)
        f.write('\n\n')


print("finish")