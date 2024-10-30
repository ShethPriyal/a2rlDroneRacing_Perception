from setuptools import setup

package_name = 'perception'  # replace with your package name

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=[
        'setuptools',
        'opencv-python',  # Add OpenCV if you're using cv2
        'torch',          # Add PyTorch if you're using it
        'torchvision',    # Add if you need torchvision (optional)
        'cv_bridge',      # Ensure cv_bridge is listed for image handling
    ],
    zip_safe=True,
    maintainer='priyal',
    maintainer_email='pnsheth@umich.edu',
    description='A package for YOLOv5 inference in ROS 2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolov5_inference_node = perception.detector_node:main',  # Update with your actual module path
        ],
    },
)

