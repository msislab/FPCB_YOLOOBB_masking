# from ultralytics import YOLO
import cv2
import numpy as np

# Load the image
image_path = 'ref_0_0dky55bw_m.png'
image = cv2.imread(image_path)
image_copy = image.copy()

# Initialize variables``
drawing = False
start_point = (-1, -1)
end_point = (-1, -1)

x_shape, y_shape = image.shape[1], image.shape[0]



# Define the callback function
def draw_rectangle(event, x, y, flags, param):
    global start_point, end_point, drawing, image_copy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            image_copy = image.copy()
            cv2.rectangle(image_copy, start_point, (x, y), (255, 0, 0), 2)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        cv2.rectangle(image_copy, start_point, end_point, (255, 0, 0), 2)
        process_and_rotate_rectangle()

def process_and_rotate_rectangle():
    global start_point, end_point, image_copy

    # Extract the ROI
    x1, y1 = start_point
    x2, y2 = end_point
    roi = image[y1:y2, x1:x2]

    # Apply thresholding
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh_roi = cv2.threshold(gray_roi, 100, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded ROI
    contours, _ = cv2.findContours(thresh_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the thresholded ROI
    contour_image = cv2.cvtColor(thresh_roi, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    yolo_obb_labels = []

    for contour in contours:
        # Fit the minimum area rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)  # Convert to integer
        
        # Get the coordinates of the bounding box
        x_1, y_1 = box[0]
        x_2, y_2 = box[1]
        x_3, y_3 = box[2]
        x_4, y_4 = box[3]
        
        # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
        yolo_obb_label = f"{x_1/x_shape} {y_1/y_shape} {x_2/x_shape} {y_2/y_shape} {x_3/x_shape} {y_3/y_shape} {x_4/x_shape} {y_4/y_shape}"
        print(yolo_obb_label)
        yolo_obb_labels.append(yolo_obb_label)

    cv2.line(contour_image, (x_1,y_1), (x_2, y_2), (0, 0, 255), 3)
    cv2.line(contour_image, (x_2,y_2), (x_3, y_3), (0, 0, 255), 3)
    cv2.line(contour_image, (x_3,y_3), (x_4, y_4), (0, 0, 255), 3)
    cv2.line(contour_image, (x_4,y_4), (x_1, y_1), (0, 0, 255), 3)

    # Display the processed ROI in a new window
    cv2.imshow('Processed ROI', contour_image)


# Create a window and set the mouse callback function
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw_rectangle)

while True:
    cv2.imshow('Image', image_copy)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cv2.destroyAllWindows()