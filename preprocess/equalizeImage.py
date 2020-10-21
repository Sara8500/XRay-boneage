import numpy as np
import argparse
import cv2
import os

"""
 source: https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
"""


def equalize_histogram(file):
    img = cv2.imread(file, 0)
    equalized_image = cv2.equalizeHist(img)
    return equalized_image


def equalize_clahe(file):
    # create a CLAHE object (Arguments are optional).
    img = cv2.imread(file, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(img)
    return equalized_image


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the image file")
    args = vars(ap.parse_args())

    res_hist = equalize_histogram(args["image"])
    res_clahe = equalize_clahe(args["image"])

    original_image = cv2.imread(args["image"], 0)
    demo_image_hist = np.hstack((original_image, res_hist))  # stacking images side-by-side
    demo_image_clahe = np.hstack((original_image, res_clahe))  # stacking images side-by-side

    if not os.path.isdir("./tmp"):
        os.mkdir("./tmp")

    cv2.imwrite('tmp/equalized_image_hist.png', demo_image_hist)
    cv2.imwrite('tmp/equalized_image_clahe.png', demo_image_clahe)