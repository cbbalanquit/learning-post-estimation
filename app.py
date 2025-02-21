import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="YOLO Pose Detection",
    page_icon="🧍",
    layout="wide"
)

# Sidebar
st.sidebar.title("YOLO Pose Detection")
st.sidebar.markdown("---")

# Model selection
model_options = {
    "YOLO-Pose Nano": "yolov8n-pose.pt",
    "YOLO-Pose Small": "yolov8s-pose.pt",
    "YOLO-Pose Medium": "yolov8m-pose.pt",
    "YOLO-Pose Large": "yolov8l-pose.pt"
}
selected_model = st.sidebar.selectbox("Select YOLO Model", list(model_options.keys()))

# Confidence threshold
confidence = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.5, 0.05)

# Main content
st.title("Skeleton Keypoint Detection with YOLO")

tab1, tab2, tab3 = st.tabs(["Webcam", "Upload Image", "Upload Video"])

# Function to process frames with YOLO
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

# Load the selected model
model = load_model(model_options[selected_model])

with tab1:
    st.header("Webcam Detection")
    
    run_webcam = st.button("Start Webcam")
    stop_webcam = st.button("Stop Webcam")
    
    frame_placeholder = st.empty()
    
    if run_webcam:
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened() and not stop_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run YOLO detection
            results = model.predict(frame_rgb, conf=confidence, verbose=False)
            
            # Get the annotated frame
            annotated_frame = results[0].plot()
            
            # Display the annotated frame
            frame_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)
        
        # Release the webcam when stopped
        cap.release()

with tab2:
    st.header("Upload Image")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert the uploaded file to an image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Run YOLO detection
        results = model.predict(image_np, conf=confidence, verbose=False)
        
        # Display original and processed images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image_np, channels="RGB", use_column_width=True)
        
        with col2:
            st.subheader("Detected Keypoints")
            annotated_img = results[0].plot()
            st.image(annotated_img, channels="RGB", use_column_width=True)
        
        # Display detection information
        if len(results[0].boxes) > 0:
            st.success(f"Detected {len(results[0].boxes)} person(s)")
            
            # Show keypoints confidence
            if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                st.subheader("Keypoint Confidence")
                keypoints = results[0].keypoints.data
                if len(keypoints) > 0:
                    kp = keypoints[0]  # Take the first person's keypoints
                    kp_conf = kp[:, 2].tolist()  # Confidence values
                    
                    keypoint_names = [
                        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                        "left_wrist", "right_wrist", "left_hip", "right_hip",
                        "left_knee", "right_knee", "left_ankle", "right_ankle"
                    ]
                    
                    # Create a bar chart of keypoint confidences
                    st.bar_chart({
                        "Confidence": kp_conf
                    })
                    
                    # Display a table with keypoint names and confidences
                    conf_data = {"Keypoint": keypoint_names, "Confidence": [f"{conf:.2f}" for conf in kp_conf]}
                    st.dataframe(conf_data)
        else:
            st.warning("No persons detected in the image.")

with tab3:
    st.header("Upload Video")
    
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(uploaded_video.read())
        temp_file_path = temp_file.name
        temp_file.close()
        
        # Video options
        process_video = st.button("Process Video")
        video_placeholder = st.empty()
        
        if process_video:
            # Open the video file
            cap = cv2.VideoCapture(temp_file_path)
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Create a temporary file for the processed video
            output_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Get total frame count
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            processed_frames = 0
            
            # Process the video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run YOLO detection
                results = model.predict(frame_rgb, conf=confidence, verbose=False)
                
                # Get the annotated frame
                annotated_frame = results[0].plot()
                
                # Convert back to BGR for saving
                annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                
                # Write the frame to the output video
                out.write(annotated_frame_bgr)
                
                # Update progress
                processed_frames += 1
                progress = processed_frames / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {processed_frames}/{total_frames}")
            
            # Release resources
            cap.release()
            out.release()
            
            # Show the processed video
            status_text.text("Video processing complete!")
            video_file = open(output_path, 'rb')
            video_bytes = video_file.read()
            video_placeholder.video(video_bytes)
            
            # Clean up temp files
            os.unlink(temp_file_path)

# Add information about the application
st.sidebar.markdown("---")
st.sidebar.info("""
This application uses YOLO v8 from Ultralytics to detect human pose keypoints.

- Use the webcam for real-time detection
- Upload images for detailed analysis
- Process videos to track movement
""")