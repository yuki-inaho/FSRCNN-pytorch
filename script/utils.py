import cv2
import numpy as np


def show_image(title: str, image: np.ndarray):
    while True:
        cv2.imshow(title, image)
        key = cv2.waitKey(10)

        if key & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
