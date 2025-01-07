import cv2
import numpy as np

# Helper functions for processing video feed and detecting centerline
def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def canny_edge_detection(image, low_threshold=150, high_threshold=200):
    return cv2.Canny(image, low_threshold, high_threshold)

def dilate_with_buffer(image, buffer_radius=5):
    kernel = np.ones((buffer_radius, buffer_radius), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def detect_centerline(image, orientation="vertical", buffer_radius=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = apply_gaussian_blur(gray)
    edges = canny_edge_detection(blurred)
    dilated_edges = dilate_with_buffer(edges, buffer_radius)

    line_image = image.copy()

    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        relevant_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if orientation == "vertical" and abs(x2 - x1) < abs(y2 - y1):
                relevant_lines.append(((x1 + x2) // 2, abs(y2 - y1)))
            elif orientation == "horizontal" and abs(y2 - y1) < abs(x2 - x1):
                relevant_lines.append(((y1 + y2) // 2, abs(x2 - x1)))

        if relevant_lines:
            if orientation == "vertical":
                weighted_sum = sum(x * length for x, length in relevant_lines)
                total_length = sum(length for _, length in relevant_lines)
                center_x = int(weighted_sum / total_length) if total_length > 0 else image.shape[1] // 2
                cv2.line(line_image, (center_x, 0), (center_x, line_image.shape[0]), (0, 0, 255), 2)
            elif orientation == "horizontal":
                weighted_sum = sum(y * length for y, length in relevant_lines)
                total_length = sum(length for _, length in relevant_lines)
                if total_length > 0:
                    center_y = int(weighted_sum / total_length)
                else:
                    center_y = image.shape[0] // 2
                cv2.line(line_image, (0, center_y), (line_image.shape[1], center_y), (0, 0, 255), 2)

    return line_image

# Resize function to ensure all frames are the same size
def resize_frame(frame):
    return cv2.resize(frame, (1280, 720))

# Function to detect orientation of lines (horizontal or vertical)
def detect_lines_orientation(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = canny_edge_detection(gray)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    vertical_lines = 0
    horizontal_lines = 0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < abs(y2 - y1):
                vertical_lines += 1
            elif abs(y2 - y1) < abs(x2 - x1):
                horizontal_lines += 1

    if vertical_lines > horizontal_lines:
        return "vertical"
    else:
        return "horizontal"

# Function to draw parallel lines (vertical or horizontal) on the image
def draw_parallel_lines(image, orientation="vertical"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = apply_gaussian_blur(gray)
    edges = canny_edge_detection(blurred)
    dilated_edges = dilate_with_buffer(edges)

    line_image = image.copy()

    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if orientation == "vertical" and abs(x2 - x1) < abs(y2 - y1):
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green for vertical lines
            elif orientation == "horizontal" and abs(y2 - y1) < abs(x2 - x1):
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw blue for horizontal lines

    return line_image

# Video streaming function
def main():
    cap = cv2.VideoCapture(0)  # Use the first available camera
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_resized = resize_frame(frame)

        # Calculate the center of the frame
        frame_height, frame_width = frame_resized.shape[:2]

        # Define the desired width and height for the cropping box
        box_width = 400  # Adjust the width as needed
        box_height = 400  # Adjust the height as needed

        # Calculate the top-left corner for centering the crop
        x = (frame_width - box_width) // 2
        y = (frame_height - box_height) // 2

        # Crop the frame with the new, larger, and centered box
        cropped_frame = frame_resized[y:y + box_height, x:x + box_width]

        # Check if the lines in the image are more horizontal or vertical
        orientation = detect_lines_orientation(cropped_frame)

        # Draw parallel lines based on the orientation
        if orientation == "horizontal":
            processed_frame = draw_parallel_lines(cropped_frame, orientation="horizontal")
            processed_frame = detect_centerline(processed_frame, orientation="horizontal")
        else:
            processed_frame = draw_parallel_lines(cropped_frame, orientation="vertical")
            processed_frame = detect_centerline(processed_frame, orientation="vertical")

        # Show the processed frame
        cv2.imshow('Video Stream', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
