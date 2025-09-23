import cv2
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

strokes=['Butterfly','Backstroke','Breaststroke','Freestyle']
views=['Aerial','Front','Side_above_water','Side_underwater','Side_water_level']
bodies=['Swimmer_Skin_0,25_Muscle_2','Swimmer_Skin_0,25_Muscle_8','Swimmer_Skin_0,75_Muscle_2','Swimmer_Skin_0,75_Muscle_8']
waters=['Water_Quantity_0,25_Height_0,6','Water_Quantity_0,25_Height_1','data/raw/labels/Butterfly/Aerial/Swimmer_Skin_0,25_Muscle_2/Water_Quantity_0,25_Height_1,5','Water_Quantity_0,75_Height_0,6','Water_Quantity_0,75_Height_1','Water_Quantity_0,75_Height_1,5']
lightings=['Lighting_rotx_110_roty_190','Lighting_rotx_110_roty_280','Lighting_rotx_110_roty_360','Lighting_rotx_140_roty_190','Lighting_rotx_140_roty_280','Lighting_rotx_140_roty_360']
speeds=['Speed_2','Speed_3']
positions=['position_1,75','position_3,75']
formats=['base','body25','COCO']
views=['2D_cam','2D_pelvis','3D_cam','3D_pelvis']

def take_line(path,idx):
    with open(path,'r') as file:
        lines = file.readlines()
        line = lines[idx]
        line = line.replace(',','.').split(';')[:-1]
        frame=[[],[]]
        for i in range(0, len(line), 3):
            x = float(line[i])
            y = float(line[i+1])
            # ignore confidence: line[j+2]
            frame[0].append(x)
            frame[1].append(y)

        return np.array(frame)
        # print(frame)
        # print(len(frame))


def parse_data():
    pass

path_data = 'data/raw/labels/Backstroke/Aerial/Swimmer_Skin_0,25_Muscle_2/Water_Quantity_0,25_Height_0,6/Lighting_rotx_110_roty_190/Speed_2/position_1,75/COCO/2D_cam.txt'

def plot_animation(path_data):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 800)
    frame0 = take_line(path_data, 1)
    scatter = ax.scatter(frame0[0], frame0[1])
    def update(i):
        frame = take_line(path_data, i)
        scatter.set_offsets(np.c_[frame[0], frame[1]])
        return scatter,
    ani=FuncAnimation(fig,update,frames=range(1, 200),interval=50)

    plt.show()

plot_animation(path_data)

# path_video="C:/Users/lihar/Code/swim-stroke-classifier/data/raw/5 Second Countdown HD.mp4"
# path_image="C:/Users/lihar/Code/swim-stroke-classifier/data/processed"
# #extract_frames(path_video,path_image)

# data = np.load("C:/Users/lihar/Code/swim-stroke-classifier/data/raw/freestyle/freestyle_1_poses.npz")
# print('arrays in file',data.files)
# poses = data['poses']
# print(poses.shape)
# print(poses[0])