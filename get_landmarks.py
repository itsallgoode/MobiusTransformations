import numpy as np
import mediapipe as mp
import cv2
import os
import pandas as pd

'''
this file takes videos and extracts the pose landmarks from them, then creates a dataframe in the required format for the SPOTER transformer model

'''
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

columns = [
    "nose_X", "nose_Y", "leftEye_X", "leftEye_Y", "rightEye_X", "rightEye_Y", "leftEar_X", "leftEar_Y", 
    "rightEar_X", "rightEar_Y", "leftShoulder_X", "leftShoulder_Y", "rightShoulder_X", "rightShoulder_Y", 
    "leftElbow_X", "leftElbow_Y", "rightElbow_X", "rightElbow_Y", "leftWrist_X", "leftWrist_Y", "rightWrist_X", 
    "rightWrist_Y", "wrist_left_X", "thumbCMC_left_X", "thumbMP_left_X", "thumbIP_left_X", "thumbTip_left_X", 
    "indexMCP_left_X", "indexPIP_left_X", "indexDIP_left_X", "indexTip_left_X", "middleMCP_left_X", "middlePIP_left_X", 
    "middleDIP_left_X", "middleTip_left_X", "ringMCP_left_X", "ringPIP_left_X", "ringDIP_left_X", "ringTip_left_X", 
    "littleMCP_left_X", "littlePIP_left_X", "littleDIP_left_X", "littleTip_left_X", "wrist_left_Y", "thumbCMC_left_Y", 
    "thumbMP_left_Y", "thumbIP_left_Y", "thumbTip_left_Y", "indexMCP_left_Y", "indexPIP_left_Y", "indexDIP_left_Y", 
    "indexTip_left_Y", "middleMCP_left_Y", "middlePIP_left_Y", "middleDIP_left_Y", "middleTip_left_Y", "ringMCP_left_Y", 
    "ringPIP_left_Y", "ringDIP_left_Y", "ringTip_left_Y", "littleMCP_left_Y", "littlePIP_left_Y", "littleDIP_left_Y", 
    "littleTip_left_Y", "wrist_right_X", "thumbCMC_right_X", "thumbMP_right_X", "thumbIP_right_X", "thumbTip_right_X", 
    "indexMCP_right_X", "indexPIP_right_X", "indexDIP_right_X", "indexTip_right_X", "middleMCP_right_X", 
    "middlePIP_right_X", "middleDIP_right_X", "middleTip_right_X", "ringMCP_right_X", "ringPIP_right_X", 
    "ringDIP_right_X", "ringTip_right_X", "littleMCP_right_X", "littlePIP_right_X", "littleDIP_right_X", 
    "littleTip_right_X", "wrist_right_Y", "thumbCMC_right_Y", "thumbMP_right_Y", "thumbIP_right_Y", 
    "thumbTip_right_Y", "indexMCP_right_Y", "indexPIP_right_Y", "indexDIP_right_Y", "indexTip_right_Y", 
    "middleMCP_right_Y", "middlePIP_right_Y", "middleDIP_right_Y", "middleTip_right_Y", "ringMCP_right_Y", 
    "ringPIP_right_Y", "ringDIP_right_Y", "ringTip_right_Y", "littleMCP_right_Y", "littlePIP_right_Y", 
    "littleDIP_right_Y", "littleTip_right_Y", "neck_X", "neck_Y"
]

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):

    selected_keypoints = [0, 2, 5, 6, 7, 11, 12, 13, 14, 15, 16]
    pose_landmarks = np.array([])
    for i in selected_keypoints:
        if results.pose_landmarks:
            landmark_x = np.array([results.pose_landmarks.landmark[i].x])
            landmark_y = np.array([results.pose_landmarks.landmark[i].y])
        else:
            landmark_x = np.array([0])
            landmark_y = np.array([0])

        landmarks = np.concatenate([landmark_x, landmark_y])        
        pose_landmarks = np.append(pose_landmarks, landmarks)

    if results.pose_landmarks: # calculates position of neck
        neck_x = np.array([(results.pose_landmarks.landmark[11].x + results.pose_landmarks.landmark[12].x) / 2])
        neck_y = np.array([(results.pose_landmarks.landmark[11].y + results.pose_landmarks.landmark[12].y) / 2])
    else:
        neck_x = np.array([0])
        neck_y = np.array([0])
    #pose_x = np.array([[res.x] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(34*1)
    #pose_y = np.array([[res.y] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(34*1)
    lh_x = np.array([[res.x] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*1)
    rh_x = np.array([[res.x] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*1)
    lh_y = np.array([[res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*1)
    rh_y = np.array([[res.y] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*1)

    all_keypoints = np.concatenate([pose_landmarks, lh_x, lh_y, rh_x, rh_y, neck_x, neck_y])
    all_keypoints = np.around(all_keypoints, decimals=6)
    all_keypoints = np.clip(all_keypoints, 0, 1) # some coordinates were slightly higher than 1 and need to be trimmed to exactly 1
    return all_keypoints

def aggregate_dataframe(df):
    return df.agg(lambda x: '[' + ','.join(x.astype(str)) + ']').to_frame().transpose()
'''
converts each column's value to strings, joins them with commas, and wraps the result in square brackets
converts back to dataframe, transposes to one row
'''

def process_video(label, video_path):

    cap = cv2.VideoCapture(video_path)
    keypoints = []
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # Draw landmarks
            draw_landmarks(image, results)

            # Extract keypoints
            keypoints.append(extract_keypoints(results))

            # uncomment the code below if you would like to see the poses being extracted
            #cv2.imshow('Frame', image)
            #if cv2.waitKey(10) & 0xFF == ord('q'):
            #    break

    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    cap.release()
    cv2.destroyAllWindows()
    array = np.array(keypoints)
    np.around(array, decimals=6)
    df = pd.DataFrame(array, columns=columns)

    df = aggregate_dataframe(df)
    df['video_fps'] = frames
    df['video_size_height'] = height
    df['video_size_width'] = width
    df['labels'] = label

    return df

from concurrent.futures import ThreadPoolExecutor
DATA_PATH = os.path.join('WLASL100/train') # path to input videos

actions = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
def main():
    all_dataframes = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for action in actions:
            video_folder_path = os.path.join(DATA_PATH, action)
            try:
                video_files = os.listdir(video_folder_path)
            except FileNotFoundError as e:
                print(f"Directory not found: {video_folder_path}")
                continue

            for video_idx, video_file in enumerate(video_files):
                video_path = os.path.join(video_folder_path, video_file)
                futures.append(executor.submit(process_video, action, video_path))

        for future in futures:
            try:
                result = future.result()
                if isinstance(result, tuple):
                    all_dataframes.extend(result)
                else:
                    all_dataframes.append(result)
            except Exception as e:
                print(f"Error processing video: {e}")


    final_df = pd.concat(all_dataframes, ignore_index=True)
    final_df.to_csv('test.csv', index=False) 

if __name__ == "__main__":
    main()
