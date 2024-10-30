import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import torch
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# Create output directory if it doesn't exist
output_dir = 'output_images'
frame_count = 0  # Initialize this variable at the start
os.makedirs(output_dir, exist_ok=True)

class YOLOv5InferenceNode(Node):
    def __init__(self):
        super().__init__('yolov5_inference_node')
        self.subscription = self.create_subscription(
            Image,
            '/mono_py_driver/img_msg',  # Topic name for image data
            self.image_callback,
            10)
        
        self.publisher_ = self.create_publisher(String, '/yolo_inference/results', 10)
        self.br = CvBridge()
        self.model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')  # Load custom model

        self.get_logger().info('YOLOv5 Inference Node has been started.')

    def image_callback(self, data):
        global frame_count
        self.get_logger().info('Receiving image frame')

        # Convert ROS Image message to OpenCV format
        cv_image = self.br.imgmsg_to_cv2(data, "bgr8")

        # Run YOLOv5 inference
        results = self.model(cv_image)

        # Process results
        detections = results.xyxy[0].cpu().numpy()  # Get xyxy, confidence, and class prediction

        # Publish results
        if len(detections) > 0:
            result_str = self.format_results(detections)
            self.publisher_.publish(String(data=result_str))
        frame_count += 1  # Increment frame count
        # Optional: Display the image with bounding boxes
        #annotated_image = np.squeeze(results.render())  # Render predictions on the image
        #cv2.imshow('YOLOv5 Inference', annotated_image)
        #cv2.waitKey(1)
    # Render predictions on the image
        annotated_image = np.squeeze(results.render())

        # Save the annotated image
        output_file = os.path.join(output_dir, f'annotated_frame_{frame_count}.jpg')
        cv2.imwrite(output_file, annotated_image)
        
    def format_results(self, detections):
        results = []
        for det in detections:
            xmin, ymin, xmax, ymax, confidence, cls = det
            result = f"Class: {int(cls)}, Confidence: {confidence:.2f}, Box: [{xmin}, {ymin}, {xmax}, {ymax}]"
            results.append(result)
        return "\n".join(results)

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv5InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

