import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR"

def preprocess_image(image_path):
    image = cv2.imread("car 1.jpg")
    if image is None:
        print("Error: Image not found!")
        return None, None, None, None
    print("Step 1: Image loaded successfully")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Bilateral Filtering
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Apply Binary Thresholding
    _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Step 2: Preprocessing completed")
    
    return image, gray, filtered, binary

def detect_edges(binary):
    edges = cv2.Canny(binary, 50, 200)
    print("Step 3: Edge detection completed")
    return edges

def find_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("Step 4: Contours found:", len(contours))
    return sorted(contours, key=cv2.contourArea, reverse=True)[:10]

def detect_plate(image, contours):
    plate_contour = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # Looking for rectangular contours
            plate_contour = approx
            break
    
    if plate_contour is not None:
        x, y, w, h = cv2.boundingRect(plate_contour)
        plate = image[y:y+h, x:x+w]
        print("Step 5: Plate detected, extracting...")
        cv2.imshow("Extracted Plate", plate)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return plate, plate_contour
    print("No plate detected.")
    return None, None

def apply_hough_transform(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=5)
    return lines

def extract_plate_number(plate):
    gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    _, binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print("Performing OCR on extracted plate...")
    cv2.imshow("Binary Plate", binary_plate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    text = pytesseract.image_to_string(binary_plate, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    print("Raw OCR output:", text)
    return text.strip()

def main(image_path):
    image, gray, filtered, binary = preprocess_image(image_path)
    if image is None:
        return
    edges = detect_edges(binary)
    contours = find_contours(edges)
    plate, plate_contour = detect_plate(image, contours)
    
    if plate is not None:
        plate_number = extract_plate_number(plate)
        print("Detected Plate Number:", plate_number)
        
        # Draw detected plate contour
        cv2.drawContours(image, [plate_contour], -1, (0, 255, 0), 3)
        cv2.imshow("Detected Plate", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No plate detected.")

# Example usage
# main('car3.jpg')
