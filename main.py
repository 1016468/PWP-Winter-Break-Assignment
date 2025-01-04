import cv2
import numpy as np

# Helper function to apply Gaussian Blur
def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

# Function to perform Canny edge detection
def canny_edge_detection(image, low_threshold=150, high_threshold=200):
    return cv2.Canny(image, low_threshold, high_threshold)

# Function to dilate image for edge clarity
def dilate_with_buffer(image, buffer_radius=5):
    kernel = np.ones((buffer_radius, buffer_radius), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def detect_centerline(image, orientation="vertical", buffer_radius=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = apply_gaussian_blur(gray)
    edges = canny_edge_detection(blurred)
    dilated_edges = dilate_with_buffer(edges, buffer_radius)

    # Draw detected edges on a copy of the original image
    line_image = image.copy()

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        # Filter and group lines based on orientation
        relevant_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if orientation == "vertical" and abs(x2 - x1) < abs(y2 - y1):  # Vertical lines
                relevant_lines.append(((x1 + x2) // 2, abs(y2 - y1)))  # Center X and length
            elif orientation == "horizontal" and abs(y2 - y1) < abs(x2 - x1):  # Horizontal lines
                relevant_lines.append(((y1 + y2) // 2, abs(x2 - x1)))  # Center Y and length

        # Calculate and draw the centerline
        if relevant_lines:
            # Weighted average to account for line lengths
            if orientation == "vertical":
                weighted_sum = sum(x * length for x, length in relevant_lines)
                total_length = sum(length for _, length in relevant_lines)
                center_x = int(weighted_sum / total_length) if total_length > 0 else image.shape[1] // 2
                cv2.line(line_image, (center_x, 0), (center_x, line_image.shape[0]), (0, 0, 255), 2)  # Red centerline
            elif orientation == "horizontal":
                weighted_sum = sum(y * length for y, length in relevant_lines)
                total_length = sum(length for _, length in relevant_lines)
                center_y = int(weighted_sum / total_length) if total_length > 0 else image.shape[0] // 2
                cv2.line(line_image, (0, center_y), (line_image.shape[1], center_y), (0, 0, 255), 2)  # Red centerline

    return line_image


# Main function to process an image
def process_image(image_path):
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    # Detect vertical centerline on the original image
    processed_vertical = detect_centerline(image, orientation="vertical")

    # Rotate the image 90 degrees counterclockwise
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Detect horizontal centerline on the rotated image
    processed_horizontal = detect_centerline(rotated_image, orientation="horizontal")

    # Save the processed images
    vertical_image_path = 'vertical_centerline_' + image_path.split('/')[-1]
    horizontal_image_path = 'horizontal_centerline_' + image_path.split('/')[-1]
    cv2.imwrite(vertical_image_path, processed_vertical)
    cv2.imwrite(horizontal_image_path, processed_horizontal)
    print(f"Vertical centerline image saved as: {vertical_image_path}")
    print(f"Horizontal centerline image saved as: {horizontal_image_path}")

    # Display the final images
    cv2.imshow("Original with Vertical Centerline", processed_vertical)
    cv2.imshow("Rotated with Horizontal Centerline", processed_horizontal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Specify the path to your input image
    image_path = '/Users/tookd.photo/Documents/IMG_2708.jpg'  # Change this to the path of your image
    process_image(image_path)
