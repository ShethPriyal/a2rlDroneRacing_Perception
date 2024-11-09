import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import torch
import cv2
import os
import numpy as np

# Set up output directory
output_dir = 'output_images'
frame_count = 0
os.makedirs(output_dir, exist_ok=True)

class YOLOv5InferenceNode(Node):
    def __init__(self):
        super().__init__('yolov5_inference_node')
        
        # Initialize ROS subscriptions and publishers
        self.subscription = self.create_subscription(
            Image,
            '/mono_py_driver/img_msg',  # Topic for incoming image data
            self.image_callback,
            10)
        
        self.publisher_ = self.create_publisher(String, '/yolo_inference/results', 10)
        self.br = CvBridge()
        
        # Load YOLOv5 model
        # Ensure you have cloned the YOLOv5 repository and have 'best.pt' in the specified path
        self.model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='github')  # Adjust 'source' as needed
        self.model_yolo.eval()  # Set model to evaluation mode
        
        # Load custom corner detection model
        # It's recommended to define a separate model class and load state_dict for better practice
        self.corner_model = torch.load('corner_model.pt', map_location=torch.device('cpu'))  # Update path and device as needed
        self.corner_model.eval()  # Set the model to evaluation mode

        self.get_logger().info('YOLOv5 and Corner Detection Inference Node has been started.')

    def image_callback(self, data):
        global frame_count
        self.get_logger().info('Receiving image frame')

        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.br.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # YOLOv5 inference
        yolo_results = self.model_yolo(cv_image)
        yolo_detections = yolo_results.xyxy[0].cpu().numpy()  # Get [xmin, ymin, xmax, ymax, confidence, class]

        # Process each detection and run corner detection on the cropped image
        corner_predictions = []
        for det in yolo_detections:
            xmin, ymin, xmax, ymax, confidence, cls = det
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            
            # Crop ROI with boundary checks
            cropped_img = self.crop_image(cv_image, xmin, ymin, xmax, ymax)
            if cropped_img is None:
                continue  # Skip if cropping failed

            # Run corner detection on the cropped image
            corner_prediction = self.predict_corners(cropped_img)
            if corner_prediction is not None:
                corner_predictions.append({
                    'bbox': [xmin, ymin, xmax, ymax],
                    'confidence': confidence,
                    'class': int(cls),
                    'corners': corner_prediction
                })

        # Format and publish results
        result_str = self.format_results(corner_predictions)
        self.publisher_.publish(String(data=result_str))

        # Annotate and save image
        frame_count += 1
        annotated_image = self.annotate_image(cv_image, corner_predictions)
        output_file = os.path.join(output_dir, f'annotated_frame_{frame_count}.jpg')
        cv2.imwrite(output_file, annotated_image)

    def crop_image(self, image, xmin, ymin, xmax, ymax):
        """Crop the image with boundary checks."""
        height, width, _ = image.shape
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(width, xmax)
        ymax = min(height, ymax)
        if xmax <= xmin or ymax <= ymin:
            self.get_logger().warning('Invalid bounding box for cropping.')
            return None
        return image[ymin:ymax, xmin:xmax]

    def predict_corners(self, crop_img):
        """Run corner detection model on cropped image and return predicted corner points."""
        input_tensor = self.preprocess_image(crop_img)
        if input_tensor is None:
            return None

        with torch.no_grad():
            output = self.corner_model(input_tensor)
            # Assuming the output is a tensor of shape [1, 8] representing (x1, y1, x2, y2, x3, y3, x4, y4)
            # Adjust based on your actual model's output
            corner_coords = output.cpu().numpy().flatten()
        
        # Reshape to (4, 2) for four corners
        if corner_coords.size != 8:
            self.get_logger().warning('Unexpected corner detection output size.')
            return None
        corners = corner_coords.reshape(-1, 2)  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        return corners

    def preprocess_image(self, image):
        """Convert the cropped image to the input format expected by the corner model."""
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))  # Adjust size based on your model's requirement
            image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
            image = torch.from_numpy(image).permute(2, 0, 1)  # Convert to [C, H, W]
            image = image.unsqueeze(0)  # Add batch dimension
            return image
        except Exception as e:
            self.get_logger().error(f'Failed to preprocess image for corner detection: {e}')
            return None

    def format_results(self, corner_predictions):
        """Format the detection and corner results into a string for publishing."""
        results = []

        for det in corner_predictions:
            xmin, ymin, xmax, ymax = det['bbox']
            confidence = det['confidence']
            cls = det['class']
            corners = det['corners']
            
            yolo_result = f"YOLO - Class: {cls}, Confidence: {confidence:.2f}, Box: [{xmin}, {ymin}, {xmax}, {ymax}]"
            results.append(yolo_result)
            
            for i, (x, y) in enumerate(corners):
                corner_result = f"Corner {i+1} - X: {x:.2f}, Y: {y:.2f}"
                results.append(corner_result)

        return "\n".join(results)

    def annotate_image(self, image, corner_predictions):
        """Render YOLOv5 and corner detection results onto the image."""
        annotated_image = image.copy()

        for det in corner_predictions:
            xmin, ymin, xmax, ymax = det['bbox']
            confidence = det['confidence']
            cls = det['class']
            corners = det['corners']

            # Draw YOLO bounding box
            cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Blue box
            label = f"Class: {cls}, Conf: {confidence:.2f}"
            cv2.putText(annotated_image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 0, 0), 2)

            # Draw corners
            for i, (x, y) in enumerate(corners):
                # Scale corner coordinates back to original image size
                corner_x = int(xmin + x * (xmax - xmin))
                corner_y = int(ymin + y * (ymax - ymin))
                cv2.circle(annotated_image, (corner_x, corner_y), 5, (0, 255, 0), -1)  # Green dots
                cv2.putText(annotated_image, f"C{i+1}", (corner_x + 5, corner_y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return annotated_image

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv5InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down YOLOv5 and Corner Detection Inference Node.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

