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

def detect_centerline(image, buffer_radius=5, flipped=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = apply_gaussian_blur(gray)
    edges = canny_edge_detection(blurred)
    dilated_edges = dilate_with_buffer(edges, buffer_radius)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=50)
    line_image = image.copy()

    if lines is not None:
        line_segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate slope to identify vertical-like or horizontal-like lines
            slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Add small epsilon to avoid division by zero
            if not flipped:
                # Filter lines with steep slopes (vertical-like)
                if 0.5 < abs(slope):  
                    line_segments.append(((x1, y1), (x2, y2)))
            else:
                # For flipped paper, we want to find near-horizontal lines
                if abs(slope) < 0.5:  
                    line_segments.append(((x1, y1), (x2, y2)))

        # Sort by x-coordinates to find left and right lines
        left_lines = []
        right_lines = []
        for (x1, y1), (x2, y2) in line_segments:
            mid_x = (x1 + x2) // 2
            if mid_x < image.shape[1] // 2:
                left_lines.append(((x1, y1), (x2, y2)))
            else:
                right_lines.append(((x1, y1), (x2, y2)))

        # Average the left and right lines
        def average_line(lines):
            if not lines:
                return None
            x1_avg = int(sum(line[0][0] for line in lines) / len(lines))
            y1_avg = int(sum(line[0][1] for line in lines) / len(lines))
            x2_avg = int(sum(line[1][0] for line in lines) / len(lines))
            y2_avg = int(sum(line[1][1] for line in lines) / len(lines))
            return (x1_avg, y1_avg), (x2_avg, y2_avg)

        left_line = average_line(left_lines)
        right_line = average_line(right_lines)

        # Draw the left and right lines
        if left_line:
            cv2.line(line_image, left_line[0], left_line[1], (255, 0, 0), 2)  # Blue for left
        if right_line:
            cv2.line(line_image, right_line[0], right_line[1], (0, 255, 0), 2)  # Green for right

        # Calculate and draw centerline
        if left_line and right_line:
            # Interpolate centerline as the midpoint between the left and right lines
            for y in range(image.shape[0]):
                try:
                    if not flipped:
                        # Calculate centerline for vertical case
                        left_x = int(left_line[0][0] + (y - left_line[0][1]) * (left_line[1][0] - left_line[0][0]) / (left_line[1][1] - left_line[0][1] + 1e-6))
                        right_x = int(right_line[0][0] + (y - right_line[0][1]) * (right_line[1][0] - right_line[0][0]) / (right_line[1][1] - right_line[0][1] + 1e-6))
                    else:
                        # Calculate centerline for horizontal case (flipped paper)
                        left_y = int(left_line[0][1] + (x - left_line[0][0]) * (left_line[1][1] - left_line[0][1]) / (left_line[1][0] - left_line[0][0] + 1e-6))
                        right_y = int(right_line[0][1] + (x - right_line[0][0]) * (right_line[1][1] - right_line[0][1]) / (right_line[1][0] - right_line[0][0] + 1e-6))
                    center_x = (left_x + right_x) // 2
                    center_y = (left_y + right_y) // 2
                    cv2.circle(line_image, (center_x, center_y), 1, (0, 0, 255), -1)  # Red dots for centerline
                except Exception as e:
                    pass

    # Display results
    cv2.imshow("Final Line Detection with Perspective", line_image)
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

def main():
    cap = cv2.VideoCapture(0)  # Use the first available camera
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_resized = resize_frame(frame)

        # Dynamically calculate cropping dimensions for a smaller box with the same ratio
        frame_height, frame_width = frame_resized.shape[:2]
        crop_width = int(frame_width * 0.4)  # Reduce width to 40% of the resized frame
        crop_height = int(frame_height * 0.72)  # Reduce height proportionally to maintain the 5:9 ratio
        x_start = (frame_width - crop_width) // 2
        y_start = (frame_height - crop_height) // 2
        cropped_frame = frame_resized[y_start:y_start+crop_height, x_start:x_start+crop_width]

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

