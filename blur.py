import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import tempfile
import os
import ffmpeg._run as ffmpeg


def process_video(input_video_path, background_option='blur', color=(255, 255, 255)):
    cap = cv2.VideoCapture(input_video_path)
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = segment.process(rgb_frame)
        
        mask = results.segmentation_mask
        mask = (mask > 0.5).astype(np.uint8)
        
        if background_option == 'blur':
            blurred_frame = cv2.GaussianBlur(frame, (55, 55), 0)
            output = frame * mask[:, :, None] + blurred_frame * (1 - mask[:, :, None])
        else:
            bg_color = np.full(frame.shape, color, dtype=np.uint8)
            output = frame * mask[:, :, None] + bg_color * (1 - mask[:, :, None])
        
        if out is None:
            h, w, _ = frame.shape
            out = cv2.VideoWriter(temp_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))
        out.write(output.astype(np.uint8))
    
    cap.release()
    out.release()
    
    # Restore audio from original video
    final_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    input_audio = ffmpeg.input(input_video_path)
    input_video = ffmpeg.input(temp_video_path)
    ffmpeg.output(input_video, input_audio, final_video_path, vcodec='copy', acodec='aac', strict='experimental').run(overwrite_output=True)
    
    return final_video_path

def main():
    st.title("Video Background Blur/Replace App")
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    background_option = st.selectbox("Choose Background Option", ['blur', 'color'])
    color = (255, 255, 255) if background_option == 'color' else None
    
    if uploaded_file is not None:
        temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        if st.button("Process Video"):
            result_video_path = process_video(temp_file_path, background_option, color)
            st.video(result_video_path)
            
            with open(result_video_path, "rb") as f:
                st.download_button("Download Processed Video", f, file_name="processed_video.mp4", mime="video/mp4")

if __name__ == "__main__":
    main()
