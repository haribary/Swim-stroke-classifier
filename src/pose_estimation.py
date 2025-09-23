import cv2
import mediapipe as mp
import requests
import numpy as np
import torch
from PIL import Image
import supervision as sv
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation, infer_device
def run_pose_estimation_mediapipe(frame,out_path):
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
def run_pose_estimation_vitpose(frame,out_path):
    device = infer_device()
    image = Image.open(frame)

    # Detect humans in the image
    person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
    person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)

    inputs = person_image_processor(images=image, return_tensors="pt").to(person_model.device)

    with torch.no_grad():
        outputs = person_model(**inputs)

    results = person_image_processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
    )
    result = results[0]

    # Human label refers 0 index in COCO dataset
    person_boxes = result["boxes"][result["labels"] == 0]
    person_boxes = person_boxes.cpu().numpy()

    # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
    person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

    # Detect keypoints for each person found
    image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
    model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=device)

    inputs = image_processor(image, boxes=[person_boxes], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])
    image_pose_result = pose_results[0]

    xy = torch.stack([pose_result['keypoints'] for pose_result in image_pose_result]).cpu().numpy()
    scores = torch.stack([pose_result['scores'] for pose_result in image_pose_result]).cpu().numpy()

    key_points = sv.KeyPoints(
        xy=xy, confidence=scores
    )

    edge_annotator = sv.EdgeAnnotator(
        color=sv.Color.GREEN,
        thickness=1
    )
    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.RED,
        radius=2
    )
    annotated_frame = edge_annotator.annotate(
        scene=image.copy(),
        key_points=key_points
    )
    annotated_frame = vertex_annotator.annotate(
        scene=annotated_frame,
        key_points=key_points
    )
    #annotated_frame.save(out_path+'/ViTtest1.jpg')
    w,h=image.size
    print(f'Width={w}, Height={h}')
    return xy[0]


def pose_on_all():
    pass


# out_path="C:/Users/lihar/Code/swim-stroke-classifier/data/processed"
# frame="C:/Users/lihar/Code/swim-stroke-classifier/data/raw/WIN_20250919_00_19_59_Pro.jpg"
# run_pose_estimation_mediapipe(frame,out_path)
frame="C:/Users/lihar/Code/swim-stroke-classifier/data/raw/WIN_20250922_22_37_57_Pro.jpg"
out_path='C:/Users/lihar/Code/swim-stroke-classifier/data/processed'
run_pose_estimation_vitpose(frame,out_path)