import os
import cv2
import mediapipe as mp
import math
import numpy as np




class stroke_identifiication():
    def __init__(self, video_path="video.mp4"):
        self.video_path=video_path
        print(video_path)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
    def calculate_angle(self,a, b, c):
        """Calculates the angle between three keypoints"""
        # Convert NormalizedLandmark objects to tuples
        a = (a.x, a.y)
        b = (b.x, b.y)
        c = (c.x, c.y)
        
        # Calculate the angle using the dot product and arc cosine
        radians = np.arccos(
            np.dot(np.array(b) - np.array(a), np.array(c) - np.array(b))
            / (np.linalg.norm(np.array(b) - np.array(a)) * np.linalg.norm(np.array(c) - np.array(b)))
        )
        return np.degrees(radians)

    
    def angle_between_points(self,a, b, c, d):
        vector1 = (b.x - a.x, b.y - a.y, b.z - a.z)
        vector2 = (d.x - c.x, d.y - c.y, d.z - c.z)
        dot_product = sum([v1 * v2 for v1, v2 in zip(vector1, vector2)])
        magnitude1 = sum([v1 ** 2 for v1 in vector1]) ** 0.5
        magnitude2 = sum([v2 ** 2 for v2 in vector2]) ** 0.5
        cos_theta = dot_product / (magnitude1 * magnitude2)
        angle_rad = math.acos(cos_theta)
        angle_deg = math.degrees(angle_rad)
        return angle_deg
        
    
    def detect_backstroke(self, pose_landmarks):
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

    def detect_front_crawl(self,pose_landmarks):
        left_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_elbow = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]
        left_wrist = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
        
        # Calculate the angles between the shoulder, elbow, and wrist
        left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
    
        right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Perform stroke correction based on the angles
        if left_angle >160 or right_angle >160:
            return True
        else:
            return False



    def detect_butterfly_stroke(self,pose_landmarks):
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        left_wrist_above_shoulder = left_wrist.y < left_shoulder.y
        right_wrist_above_shoulder = right_wrist.y < right_shoulder.y
        left_elbow_above_shoulder = left_elbow.y < left_shoulder.y
        right_elbow_above_shoulder = right_elbow.y < right_shoulder.y
        left_wrist_shoulder_distance = math.sqrt(
            (left_wrist.x - left_shoulder.x) ** 2 + (left_wrist.y - left_shoulder.y) ** 2
        )
        right_wrist_shoulder_distance = math.sqrt(
            (right_wrist.x - right_shoulder.x) ** 2 + (right_wrist.y - right_shoulder.y) ** 2
        )
        left_wrist_hip_distance = math.sqrt(
            (left_wrist.x - left_hip.x) ** 2 + (left_wrist.y - left_hip.y) ** 2
        )
        right_wrist_hip_distance = math.sqrt(
            (right_wrist.x - right_hip.x) ** 2 + (right_wrist.y - right_hip.y) ** 2
        )
        
        if (
            left_wrist_above_shoulder
            and right_wrist_above_shoulder
            and left_elbow_above_shoulder
            and right_elbow_above_shoulder
            and left_wrist_shoulder_distance > 0.4
            and right_wrist_shoulder_distance > 0.4
            and left_wrist_hip_distance > 0.4
            and right_wrist_hip_distance > 0.4
        ):
            return True
        else:
            return False




    def analyze_video(self,video_path):
        strokes = []
        stroke_styles = []
        stroke_count = 0
        max_consecutive_stroke_frames = 120
        consecutive_frames = 0
        prev_stroke_style = None
        stroke_style_final = False
        stroke_style_count_dict = {}
        original_position_left_wrist = None
        original_position_right_wrist = None
        stroke_flag = False
        thickness = 3
        
        video_capture = cv2.VideoCapture(video_path)
    
        with self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while video_capture.isOpened():
                success, frame = video_capture.read()
    
                if not success:
                    break
    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
    
                if results.pose_landmarks is not None:
                    left_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    left_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
                    right_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
    
                    left_elbow = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
                    right_elbow = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
    
                    left_knee = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
                    right_knee = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
                    left_ankle = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
                    right_ankle = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
    
                    left_distance_stroke = ((left_shoulder.x - left_wrist.x) ** 2 + (left_shoulder.y - left_wrist.y) ** 2) ** 0.5
                    right_distance_stroke = ((right_shoulder.x - right_wrist.x) ** 2 + (right_shoulder.y - right_wrist.y) ** 2) ** 0.5
                    angle = self.angle_between_points(left_shoulder, right_shoulder, left_wrist, right_wrist)                 
    
                    backstroke_detected = self.detect_backstroke(results.pose_landmarks)
                    front_crawl_detected = self.detect_front_crawl(results.pose_landmarks)
                    butterfly_detected = self.detect_butterfly_stroke(results.pose_landmarks)
    
                    if consecutive_frames < max_consecutive_stroke_frames:
                        consecutive_frames += 1
                    else:
                        stroke_style_final = True
    
                    if angle < 5 or butterfly_detected and not backstroke_detected :
                        stroke_style = 'Butterfly Stroke'
                    elif backstroke_detected and angle:
                        stroke_style = 'Backstroke'
                    elif front_crawl_detected or angle>100:
                        stroke_style = 'Front Crawl'
                    else:
                        stroke_style = 'None'
    
                    if stroke_style != 'None':
                        stroke_styles.append(stroke_style)
    
    
                    if stroke_style_final:
                        stroke_style_count_dict = {s: stroke_styles.count(s) for s in set(stroke_styles)}
                        break
    
                    cv2.putText(frame, f'Stroke Style: {stroke_style}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
                self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                #cv2.imshow('Swimming Annotation', frame)
    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
            video_capture.release()
            cv2.destroyAllWindows()
        
        return stroke_style_count_dict
    def final_stroke(self, stroke_style_count_dict=None ):
        stroke_style_max="No stroke identified, fix the camera angle or correct swim posture"
        if stroke_style_count_dict!={}:
            stroke_style_max = max(stroke_style_count_dict, key=stroke_style_count_dict.get)
            
        else:
            stroke_style_count_dict={}
        return stroke_style_max

