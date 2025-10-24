import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR"

# Load Image
image = cv2.imread("car3.jpg")

# Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Bilateral Filtering (Reduces noise while keeping edges sharp)
bilateral = cv2.bilateralFilter(gray, 11, 17, 17)

# Convert to Binary using Adaptive Thresholding
binary = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)

# Apply Edge Detection
edges = cv2.Canny(binary, 50, 150)

# Find Contours
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Loop through contours to detect the possible number plate
plate = None
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    if len(approx) == 4:  # Possible number plate shape
        x, y, w, h = cv2.boundingRect(cnt)

        # Min and Max points
        min_point = (x, y)  # Top-left
        max_point = (x + w, y + h)  # Bottom-right

        print(f"Min Point: {min_point}")
        print(f"Max Point: {max_point}")

        # Crop the plate
        plate = binary[y:y+h, x:x+w]

        # Draw bounding box around plate
        cv2.rectangle(image, min_point, max_point, (0, 255, 0), 2)
        break  # Assuming only one plate

# OCR to Extract Text from Plate
if plate is not None:
    text = pytesseract.image_to_string(plate, config='--psm 8')
    print("Detected Plate Number:", text)

    # Display Processed Plate
    cv2.imshow("Extracted Plate", plate)

# Show Results
cv2.imshow("Original Image", image)
cv2.imshow("Grayscale", gray)
cv2.imshow("Bilateral Filter", bilateral)
cv2.imshow("Binary Image", binary)
cv2.imshow("Edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
