import cv2
import numpy as np

# Load the Haar Cascade for number plate detection
numberPlateCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

# Read the input image
img = cv2.imread('car12.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding to the grayscale image
_, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

# Apply bilateral filter to reduce noise while keeping edges sharp
bilateral = cv2.bilateralFilter(img, 9, 75, 75)

# Convert the bilateral-filtered image to grayscale for edge detection
gray_bilateral = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)

# Use Canny edge detection
edges = cv2.Canny(gray_bilateral, 100, 200)

# Use Hough Line Transform to detect lines on the edges (Optional, not strictly needed for refining bounding box)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

# Draw the detected lines (if any)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Detect plates using the Haar Cascade classifier
plates = numberPlateCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(25, 25))

# Loop through detected plates and refine their bounding boxes using contours
for (x, y, w, h) in plates:
    # Crop the region of interest (ROI) for further processing
    plate_roi = img[y:y + h, x:x + w]

    # Convert the ROI to grayscale and apply thresholding
    plate_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    _, plate_bin = cv2.threshold(plate_gray, 120, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(plate_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and sort the contours based on area and shape
    for contour in contours:
        # Get the bounding box for the contour
        x_contour, y_contour, w_contour, h_contour = cv2.boundingRect(contour)

        # Apply an aspect ratio filter if needed to match plate dimensions
        aspect_ratio = float(w_contour) / h_contour
        if aspect_ratio > 2 and aspect_ratio < 6:  # Assuming the aspect ratio of a plate is between 2 and 6
            # Draw a refined rectangle around the contour
            cv2.rectangle(img, (x + x_contour, y + y_contour), (x + x_contour + w_contour, y + y_contour + h_contour),
                          (0, 255, 0), 2)

# Display the final detection with refined bounding boxes
cv2.imshow('Plate Detection with Refined Box', img)

# Wait for key press to close all windows
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
