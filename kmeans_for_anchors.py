import os
import numpy as np

def load_annotations(label_dir, image_width, image_height):
    """
    Load bounding box annotations and convert them to pixel coordinates.

    Args:
        label_dir (str): Path to the directory containing label files in YOLO format.
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.

    Returns:
        np.ndarray: An array of bounding box widths and heights in pixel format.
    """
    annotations = []
    class_mapping = {}  # 用于映射标签名称到数值
    current_class_id = 0

    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            label_path = os.path.join(label_dir, label_file)

            with open(label_path, 'r') as file:
                for line in file:
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"Skipping invalid line in {label_file}: {line.strip()}")
                        continue
                    
                    class_name = parts[0]  # 标签名称
                    if class_name not in class_mapping:
                        class_mapping[class_name] = current_class_id
                        current_class_id += 1
                    
                    # 提取宽高并转为像素
                    _, center_x, center_y, width, height = parts
                    width = float(width) * image_width
                    height = float(height) * image_height

                    annotations.append([width, height])

    print(f"Class mapping: {class_mapping}")
    return np.array(annotations)



def kmeans_anchors(boxes, k=9, max_iter=1000, eps=1e-4):
    """
    Perform k-means clustering to generate anchors.

    Args:
        boxes (np.ndarray): Array of bounding box widths and heights.
        k (int): Number of clusters (anchors).
        max_iter (int): Maximum number of iterations for k-means.
        eps (float): Convergence threshold.

    Returns:
        np.ndarray: The k anchor boxes.
    """
    # Initialize clusters as k random boxes
    indices = np.random.choice(boxes.shape[0], k, replace=False)
    clusters = boxes[indices]

    for _ in range(max_iter):
        distances = 1 - (np.minimum(boxes[:, None, :], clusters[None, :, :]).prod(axis=2) /
                         np.maximum(boxes[:, None, :], clusters[None, :, :]).prod(axis=2))

        nearest_clusters = np.argmin(distances, axis=1)

        new_clusters = np.array([boxes[nearest_clusters == i].mean(axis=0) for i in range(k)])

        if np.all(np.abs(new_clusters - clusters) < eps):
            break

        clusters = new_clusters

    return clusters


def generate_anchors(label_dir, image_width, image_height, num_anchors=9):
    """
    Generate YOLO anchors using k-means clustering.

    Args:
        label_dir (str): Path to the directory containing label files in YOLO format.
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.
        num_anchors (int): Number of anchors to generate.

    Returns:
        np.ndarray: The generated anchors sorted by area.
    """
    annotations = load_annotations(label_dir, image_width, image_height)

    # Perform k-means clustering
    anchors = kmeans_anchors(annotations, k=num_anchors)

    # Sort anchors by area
    anchors = anchors[np.argsort(anchors.prod(axis=1))]
    return anchors


# Example usage
label_dir = "Dataset1.0/labels"  # Path to the labels
image_width = 1920  # Replace with actual image width
image_height = 1080  # Replace with actual image height
num_anchors = 9  # Number of anchors to generate

anchors = generate_anchors(label_dir, image_width, image_height, num_anchors)
print("Generated anchors (width, height):")
print(anchors)
print("Normalized anchors (to [0, 1]):")
print(anchors / [image_width, image_height])
