import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR"


#load the image

image = cv2.imread('car 1.jpg')

# convert the image to grayscale

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply Gaussian blur

blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# apply thresholding

thresh_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# dilate the image to remove noise

dilated_image = cv2.dilate(thresh_image, np.ones((3, 3), np.uint8), iterations=1)

# find contours

contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# sort contours by area

contours = sorted(contours, key=cv2.contourArea, reverse=True)

# initialize variables for the plate and bounding box

plate = None
bounding_box = None

# loop through contours

for contour in contours:

    # approximate the contour shape

    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    # check if the approximated contour has 4 points
    if len(approx) == 4:
        # compute the bounding box
        bounding_box = cv2.boundingRect(approx)
        # compute the area of the bounding box
        area = bounding_box[2] * bounding_box[3]
        # compute the aspect ratio
        aspect_ratio = bounding_box[2] / bounding_box[3]
        # check if the area is large enough and the aspect ratio is acceptable
        if area > 1000 and aspect_ratio > 3 and aspect_ratio < 7:
            # crop the plate
            plate = gray_image[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]]
            break

        # display the original image, threshold image, and plate
        cv2.imshow('Original Image', image)
        cv2.imshow('Threshold Image', thresh_image)
        cv2.imshow('Plate', plate)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # apply OCR to the plate
        text = pytesseract.image_to_string(plate)
        print('Detected Plate Number:', text)
        break

    # display the original image, threshold image, and bounding box
    cv2.imshow('Original Image', image)
    cv2.imshow('Threshold Image', thresh_image)
    cv2.imshow('Bounding Box', cv2.drawContours(image, [approx], -1, (0, 255, 0), 2))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
