import cv2
import os

def visualize_annotations(image_dir, label_dir, output_dir, image_width=1280, image_height=720):
    """
    Visualizes annotations by drawing bounding boxes on images.

    Args:
        image_dir (str): Path to the directory containing images.
        label_dir (str): Path to the directory containing label files in YOLO format.
        output_dir (str): Path to save visualized images.
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.
    """
    os.makedirs(output_dir, exist_ok=True)

    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            base_name = os.path.splitext(label_file)[0]
            image_path = os.path.join(image_dir, f"{base_name}.jpg")
            label_path = os.path.join(label_dir, label_file)
            output_path = os.path.join(output_dir, f"{base_name}_visualized.jpg")

            # Ensure the image exists
            if not os.path.exists(image_path):
                print(f"Warning: Image file {image_path} does not exist. Skipping.")
                continue

            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Unable to read image {image_path}. Skipping.")
                continue

            # Read the label file and draw bounding boxes
            with open(label_path, 'r') as file:
                for line in file:
                    class_id, center_x, center_y, width, height = map(float, line.split())

                    # Convert YOLO format to pixel coordinates
                    x_min = int((center_x - width / 2) * image_width)
                    y_min = int((center_y - height / 2) * image_height)
                    x_max = int((center_x + width / 2) * image_width)
                    y_max = int((center_y + height / 2) * image_height)

                    # Draw the bounding box and class ID
                    color = (0, 255, 0)  # Green color for boxes
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(image, f"Class {int(class_id)}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Save the visualized image
            cv2.imwrite(output_path, image)
            print(f"Saved visualized image to {output_path}")

# Example usage
image_dir = "Dataset1.0/images"  # Path to the images
label_dir = "Dataset1.0/labels"  # Path to the labels
output_dir = "Dataset1.0/visualized"  # Path to save visualized images
image_width = 1920  # Replace with actual image width
image_height = 1080  # Replace with actual image height

visualize_annotations(image_dir, label_dir, output_dir, image_width, image_height)
