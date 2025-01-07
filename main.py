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

def detect_centerline_with_improvements(image, orientation="vertical", buffer_radius=5, merge_threshold=50):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = apply_gaussian_blur(gray)
    edges = canny_edge_detection(blurred)
    dilated_edges = dilate_with_buffer(edges, buffer_radius)

    debug_image = image.copy()  # For visualizing raw detected lines
    line_image = image.copy()

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        # Visualize detected lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_image, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Cyan lines for visualization

        cv2.imshow("Raw Detected Lines", debug_image)

        # Filter and process lines based on orientation
        relevant_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if orientation == "vertical" and abs(x2 - x1) < abs(y2 - y1):  # Vertical condition
                mid_x = (x1 + x2) // 2
                length = abs(y2 - y1)
                relevant_lines.append((mid_x, length))
            elif orientation == "horizontal" and abs(y2 - y1) < abs(x2 - x1):  # Horizontal condition
                mid_y = (y1 + y2) // 2
                length = abs(x2 - x1)
                relevant_lines.append((mid_y, length))

        # Merge lines that are within the merge_threshold
        merged_lines = []
        relevant_lines.sort()  # Sort by position (x or y depending on orientation)
        for pos, length in relevant_lines:
            if not merged_lines or abs(pos - merged_lines[-1][0]) > merge_threshold:
                merged_lines.append((pos, length))
            else:
                # Merge by taking the weighted average position and cumulative length
                prev_pos, prev_length = merged_lines[-1]
                total_length = prev_length + length
                new_pos = int((prev_pos * prev_length + pos * length) / total_length)
                merged_lines[-1] = (new_pos, total_length)

        # Visualize merged lines for debugging
        merged_image = line_image.copy()
        for pos, _ in merged_lines:
            if orientation == "vertical":
                cv2.line(merged_image, (pos, 0), (pos, merged_image.shape[0]), (0, 255, 0), 2)  # Green for merged lines
            elif orientation == "horizontal":
                cv2.line(merged_image, (0, pos), (merged_image.shape[1], pos), (0, 255, 0), 2)  # Green for merged lines
        cv2.imshow("Merged Lines", merged_image)

        # Calculate and draw the centerline
        if len(merged_lines) > 1:
            if orientation == "vertical":
                weighted_sum = sum(x * length for x, length in merged_lines)
                total_length = sum(length for _, length in merged_lines)
                center_x = int(weighted_sum / total_length) if total_length > 0 else image.shape[1] // 2
                cv2.line(line_image, (center_x, 0), (center_x, line_image.shape[0]), (0, 0, 255), 2)  # Red centerline
            elif orientation == "horizontal":
                weighted_sum = sum(y * length for y, length in merged_lines)
                total_length = sum(length for _, length in merged_lines)
                if total_length > 0:
                    center_y = int(weighted_sum / total_length)
                else:
                    center_y = image.shape[0] // 2
                cv2.line(line_image, (0, center_y), (line_image.shape[1], center_y), (0, 0, 255), 2)  # Red centerline

    # Display the final result
    cv2.imshow("Final Line Detection", line_image)
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

