# YOLO Pose Detection Web App

This Streamlit application demonstrates the capabilities of YOLO pose detection for identifying human skeleton keypoints in real-time using your webcam or through uploaded images and videos.

## Features

- **Webcam Detection**: Real-time pose detection using your webcam
- **Image Upload**: Upload and analyze images to detect pose keypoints
- **Video Upload**: Process videos to track pose keypoints over time
- **Model Selection**: Choose from different YOLO pose detection models (nano, small, medium, large)
- **Confidence Control**: Adjust the detection confidence threshold
- **Keypoint Analysis**: View confidence scores for each detected keypoint
- **Cross-Platform**: Docker support for running on any system

## Requirements

- Python 3.8 or higher
- Webcam (for real-time detection)
- Docker (optional, for containerized deployment)
- uv package manager (recommended) or pip

## Installation Options

### Option 1: Using uv (Recommended)

1. Install uv if not already installed:
   ```bash
   pip install uv
   ```

2. Create and activate a virtual environment:
   ```bash
   uv venv
   # Windows
   .venv\Scripts\activate
   # Mac/Linux
   source .venv/bin/activate
   ```

3. Install the project and dependencies:
   ```bash
   uv pip install -e .
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

### Option 2: Using Docker (Cross-Platform)

1. Build and run with Docker Compose:
   ```bash
   docker-compose up
   ```

2. Access the application at http://localhost:8501

#### Using GPU Acceleration (Optional)

If you have a CUDA-compatible GPU and want to use it for faster inference:

1. Install the NVIDIA Container Toolkit if not already installed
2. Uncomment the GPU-related lines in `docker-compose.yml`
3. Run with Docker Compose as shown above

### Option 3: Traditional pip (Fallback)

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

Once the application is running:

1. Open your web browser and navigate to the URL (usually http://localhost:8501)
2. Use the different tabs to test pose detection:
   - **Webcam**: Click "Start Webcam" to begin real-time pose detection
   - **Upload Image**: Upload an image for pose analysis
   - **Upload Video**: Upload a video for processing
3. Adjust the model and confidence settings in the sidebar

## Customization

- Adjust the confidence threshold in the sidebar to filter detections
- Select different YOLO models based on your performance needs:
  - Nano: Fastest but less accurate
  - Small: Good balance of speed and accuracy
  - Medium: More accurate but slower
  - Large: Most accurate but requires more computational resources

## Troubleshooting

- **Docker Camera Access**: When using Docker, you may need to grant camera access. On Linux, use `--device=/dev/video0:/dev/video0` when running the container.
- **Performance Issues**: Try using a smaller YOLO model or reducing your camera resolution.
- **Webcam Not Working**: Ensure your browser has permission to access your camera.
- **Linux Dependencies**: If running natively on Linux and encountering OpenCV errors, you may need to install additional dependencies: `sudo apt-get install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev`

## Credits

This application uses:
- [Streamlit](https://streamlit.io/) for the web interface
- [Ultralytics YOLO v8](https://github.com/ultralytics/ultralytics) for pose detection
- [OpenCV](https://opencv.org/) for image and video processing
- [uv](https://github.com/astral-sh/uv) for Python package management
- [Docker](https://www.docker.com/) for containerization