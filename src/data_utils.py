import cv2
import numpy as np
def extract_frames(path_video,path_images,fps=5):
    """Takes mp4 video and saves it as a folder of images of each frame"""
    vidcap = cv2.VideoCapture(path_video)
    original_fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames_interval = int(original_fps/fps)
    success=True
    frames=0
    saved=0
    while success:
        success,frame = vidcap.read() #success is True if frame was successfully read
        if success and frames%frames_interval == 0:
            cv2.imwrite(path_images+f"/frame{saved+1}.jpg",frame)
            saved+=1
        frames+=1
    vidcap.release()


path_video="C:/Users/lihar/Code/swim-stroke-classifier/data/raw/5 Second Countdown HD.mp4"
path_image="C:/Users/lihar/Code/swim-stroke-classifier/data/processed"
extract_frames(path_video,path_image)