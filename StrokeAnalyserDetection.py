import os
import cv2
import mediapipe as mp
import math


#Combine untitled 5 and 6
def angle_between_points(a, b, c, d):
    vector1 = (b.x - a.x, b.y - a.y, b.z - a.z)
    vector2 = (d.x - c.x, d.y - c.y, d.z - c.z)
    dot_product = sum([v1 * v2 for v1, v2 in zip(vector1, vector2)])
    magnitude1 = sum([v1 ** 2 for v1 in vector1]) ** 0.5
    magnitude2 = sum([v2 ** 2 for v2 in vector2]) ** 0.5
    cos_theta = dot_product / (magnitude1 * magnitude2)
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def detect_backstroke(pose_landmarks):
    left_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    left_elbow = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
    
    # Check the relative position and angle of the wrists and elbows to the shoulders
    left_wrist_above_shoulder = left_wrist.y < left_shoulder.y
    right_wrist_above_shoulder = right_wrist.y < right_shoulder.y
    left_elbow_above_shoulder = left_elbow.y < left_shoulder.y
    right_elbow_above_shoulder = right_elbow.y < right_shoulder.y
    left_wrist_elbow_angle = math.degrees(math.atan2(left_wrist.y - left_elbow.y, left_wrist.x - left_elbow.x))
    right_wrist_elbow_angle = math.degrees(math.atan2(right_wrist.y - right_elbow.y, right_wrist.x - right_elbow.x))
    
    if left_wrist_above_shoulder and right_wrist_above_shoulder and \
            left_elbow_above_shoulder and right_elbow_above_shoulder and \
            left_wrist_elbow_angle < 0 and right_wrist_elbow_angle < 0:
        return True
    else:
        return False

def detect_front_crawl(pose_landmarks):
    left_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    left_elbow = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
    
    # Check the relative position and angle of the wrists and elbows to the shoulders
    left_wrist_above_shoulder = left_wrist.y > left_shoulder.y
    right_wrist_above_shoulder = right_wrist.y > right_shoulder.y
    left_elbow_above_shoulder = left_elbow.y > left_shoulder.y
    right_elbow_above_shoulder = right_elbow.y > right_shoulder.y
    left_wrist_elbow_angle = math.degrees(math.atan2(left_wrist.y - left_elbow.y, left_wrist.x - left_elbow.x))
    right_wrist_elbow_angle = math.degrees(math.atan2(right_wrist.y - right_elbow.y, right_wrist.x - right_elbow.x))
    
    if left_wrist_above_shoulder and right_wrist_above_shoulder and \
            left_elbow_above_shoulder and right_elbow_above_shoulder and \
            abs(left_wrist_elbow_angle) > 150 and abs(right_wrist_elbow_angle) > 150:
        return True
    else:
        return False
    


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

input_dir = '/Users/ilesha/Downloads/Swim Strokes'

with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    print(os.listdir(input_dir))
    
    for video_file in os.listdir(input_dir):
        if video_file.endswith('Technique.mp4'):
            video_path = os.path.join(input_dir, video_file)
            video_capture = cv2.VideoCapture(video_path)
            strokes = []
            stroke_count = 0
            max_consecutive_stroke_frames = 11
            consecutive_frames=0
            prev_stroke_style = None
            stroke_styles=[]
            stroke_style_final=False
            stroke_style=None
            original_position_left_wrist =None
            original_position_right_wrist =None
            #i=0
            stroke_style='None'
            stroke_in_progress=None
            wrist_threshold = 0.05
            thickness = 3
            backstroke_detected=False
            stroke_flag=False
            
            
            while video_capture.isOpened():
                success, frame = video_capture.read()
                
                if not success:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                
                if results.pose_landmarks is not None:
                    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                    right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                    
                    left_elbow = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
                    right_elbow = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
                    
                    left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
                    right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
                    left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
                    right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
                    
                    left_distance_stroke = ((left_shoulder.x - left_wrist.x) ** 2 + (left_shoulder.y - left_wrist.y) ** 2) ** 0.5
                    right_distance_stroke = ((right_shoulder.x - right_wrist.x) ** 2 + (right_shoulder.y - right_wrist.y) ** 2) ** 0.5
                    angle = angle_between_points(left_shoulder, right_shoulder, left_wrist, right_wrist)                 
                    
                    backstroke_detected = detect_backstroke(results.pose_landmarks)
                    front_crawl_detected=detect_front_crawl(results.pose_landmarks)
                    
                    
                    
                    if consecutive_frames < max_consecutive_stroke_frames:
                        consecutive_frames+=1
                        print(consecutive_frames)
                    else:
                        stroke_style_final=True
                    print(angle)
                    if angle < 20 and not backstroke_detected:
                        stroke_style = 'Butterfly Stroke'
                        
                        
                    elif backstroke_detected:
                        stroke_style = 'Backstroke'
                    else:
                        stroke_style = 'Front Crawl'
                        
                    
                    print(stroke_style)
                    stroke_styles.append(stroke_style)
                    
                    
                    #print(left_wrist,right_wrist)
                    water_threshold = 0 
                    if left_wrist.z >= water_threshold and right_wrist.z >= water_threshold:
                        stroke_in_water = False
                    else:
                        stroke_in_water = True
                        
                    if stroke_style != prev_stroke_style:
                        if stroke_in_water ==False and not stroke_flag:
                            stroke_count += 1
                            stroke_flag = True
                    else:
                            stroke_flag = False
                        
                    prev_stroke_style = stroke_style
                    
                    # Add the stroke style to the list of strokes
                    
                    if stroke_style_final==True:
                        #print('here')
                        stroke_count_dict = {s: stroke_styles.count(s) for s in set(stroke_styles)}
                        stroke_style_max = max(stroke_count_dict, key=stroke_count_dict.get)
                        #print(stroke_style_max)
                        cv2.putText(frame, f'Stroke Style: {stroke_style_max}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:    
                        stroke_styles.append(stroke_style)
                        # Display the stroke style on the frame
                        cv2.putText(frame, f'Stroke Style: {stroke_style}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                    
                # Draw pose landmarks on the frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                # Display the stroke count on the frame
                cv2.putText(frame, f'Stroke Count: {stroke_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display the annotated frame
                cv2.imshow('Swimming Annotation', frame)
                
                # Exit the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            # Release the video capture and destroy the OpenCV windows
            video_capture.release()
            cv2.destroyAllWindows()
            