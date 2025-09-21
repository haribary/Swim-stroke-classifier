import cv2
import mediapipe as mp
def run_pose_estimation(frame,out_path):
    """"""
    mpPose = mp.solutions.pose
    mpDraw = mp.solutions.drawing_utils
    pose = mpPose.Pose(static_image_mode=True)

    img=cv2.imread(frame)
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = pose.process(imgRGB)
    if result.pose_landmarks:
        mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)
        cv2.imwrite(out_path+f"/posetest.jpg",img)
        print(result.pose_landmarks)
        return result.pose_landmarks
    print("no pose detected")
    return None

def pose_on_all():
    pass


out_path="C:/Users/lihar/Code/swim-stroke-classifier/data/processed"
frame="C:/Users/lihar/Code/swim-stroke-classifier/data/raw/WIN_20250919_00_19_59_Pro.jpg"
run_pose_estimation(frame,out_path)