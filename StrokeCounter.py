import os
import cv2
import mediapipe as mp
import math
from stroke_identification2 import stroke_identifiication as stroke_identifier

class StrokeCounter:
    def __init__(self,  video_path, stroke_detected):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.stroke_detected = stroke_detected
        self.video_path=video_path
        

    def angle_between_points(self, a, b, c, d):
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
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]

        left_wrist_above_shoulder = left_wrist.y < left_shoulder.y
        right_wrist_above_shoulder = right_wrist.y < right_shoulder.y
        left_elbow_above_shoulder = left_elbow.y < left_shoulder.y
        right_elbow_above_shoulder = right_elbow.y < right_shoulder.y
        left_wrist_elbow_angle = math.degrees(math.atan2(left_wrist.y - left_elbow.y, left_wrist.x - left_elbow.x))
        right_wrist_elbow_angle = math.degrees(math.atan2(right_wrist.y - right_elbow.y, right_wrist.x - right_elbow.x))

        if (left_wrist_above_shoulder and right_wrist_above_shoulder and
                left_elbow_above_shoulder and right_elbow_above_shoulder and
                left_wrist_elbow_angle < 0 and right_wrist_elbow_angle < 0):
            return True
        else:
            return False

    def stroke_counter_forward(self):
        print('here',self.video_path)
        with self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            
            video_capture = cv2.VideoCapture(self.video_path)
            
            strokes = []
            stroke_count = 0
            stroke_flag = False
            stroke_count = 0
            max_consecutive_stroke_frames = 10
            consecutive_strokes = 0
            prev_stroke_style = None
            stroke_styles = []
            stroke_style_final = False
            stroke_style = None
            original_position_left_wrist = None
            original_position_right_wrist = None
            i = 0
            stroke = 'None'
            stroke_in_progress = None
            wrist_threshold = 0.05
            thickness = 2

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
                    angle = self.angle_between_points(left_shoulder, right_shoulder, left_wrist, right_wrist)
                    backstroke_detected = self.detect_backstroke(results.pose_landmarks)

                    if angle < 90 and not backstroke_detected:
                        left_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
                        right_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                        left_elbow = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
                        right_elbow = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
                        left_distance_stroke = ((left_shoulder.x - left_wrist.x) ** 2 + (left_shoulder.y - left_wrist.y) ** 2) ** 0.5
                        right_distance_stroke = ((right_shoulder.x - right_wrist.x) ** 2 + (right_shoulder.y - right_wrist.y) ** 2) ** 0.5

                        if (left_elbow.y < left_shoulder.y and right_elbow.y < right_shoulder.y and
                                left_wrist.y < left_elbow.y and right_wrist.y < right_elbow.y):
                            stroke_style = 'Butterfly Stroke'
                        else:
                            stroke_style = 'Front Crawl'
                    elif backstroke_detected:
                        stroke_style = 'Back Crawl'
                    else:
                        stroke_style = 'None'

                    if consecutive_strokes <= max_consecutive_stroke_frames:
                        consecutive_strokes += 1
                        if angle < 90 and not backstroke_detected:
                            stroke_style = 'Butterfly Stroke'
                        elif backstroke_detected:
                            stroke_style = 'Backstroke'
                        else:
                            stroke_style = 'Front Crawl'
                            #stroke_styles.append(stroke_style)
                    else:
                        stroke_style_final = True

                    if original_position_left_wrist is None:
                        original_position_left_wrist = left_wrist
                        original_position_right_wrist = right_wrist


                    water_threshold = 0
                    if left_wrist.z >= water_threshold and right_wrist.z >= water_threshold:
                        stroke_in_water = False
                    else:
                        stroke_in_water = True

                    if stroke_style != prev_stroke_style:
                        if stroke_in_water == False and not stroke_flag:
                            stroke_count += 1
                            stroke_flag = True
                    else:
                        stroke_flag = False

                    prev_stroke_style = stroke_style

                    frame_height, frame_width, _ = frame.shape
                    landmarks = results.pose_landmarks.landmark
                    min_x, min_y, max_x, max_y = frame_width, frame_height, 0, 0
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    text_color = (255, 255, 255)
                    for landmark in landmarks:
                        x = int(landmark.x * frame_width)
                        y = int(landmark.y * frame_height)

                        if x < min_x:
                            min_x = x
                        if y < min_y:
                            min_y = y
                        if x > max_x:
                            max_x = x
                        if y > max_y:
                            max_y = y

                    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), thickness)

                    stroke_styles.append(stroke_style)
                    cv2.putText(frame, f'Stroke Style:  {self.stroke_detected}', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                    self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                cv2.putText(frame, f'Stroke Count: {stroke_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Swim Counter', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            video_capture.release()
            cv2.destroyAllWindows()
            
            
    def stroke_counter_topview(self):
        #print('here', self.video_path)
        with self.mp_pose.Pose(static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            video_capture = cv2.VideoCapture(self.video_path)
        
            stroke_count = 0
            max_consecutive_stroke_frames = 10
            consecutive_strokes = 0
            prev_stroke_style = None
            stroke_flag = False
            stroke_in_water_left=False
            stroke_in_water_right=False
            i=0
        
            while video_capture.isOpened():
                success, frame = video_capture.read()
                
                if not success:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                
                if results.pose_landmarks is not None:
                    left_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
                    right_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                    left_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    left_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
                    right_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                    left_elbow = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
                    right_elbow = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
                    #angle = self.angle_between_points(left_shoulder, right_shoulder, left_wrist, right_wrist)
                    stroke_in_water = False
                    water_threshold = 0
                    
                    if left_wrist.z < -0.1 and right_wrist.z > 0.06 and stroke_in_water_left==False:
                        stroke_in_water_left = True
                        stroke_in_water_right=False
                        stroke_count += 1
                        i=0
                    
                    elif right_wrist.z < -0.1 and left_wrist.z > 0.06 and stroke_in_water_right==False:
                        stroke_in_water_left = False
                        stroke_in_water_right=True
                        stroke_count += 1
                        i=0
                    
                
                cv2.putText(frame, f'Stroke Style:  {self.stroke_detected}', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)  
                self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                cv2.putText(frame, f'Stroke Count: {stroke_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Swim Counter', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            video_capture.release()
            cv2.destroyAllWindows()

#stroke_style='Butterfly Stroke'
#input_dir = '/Users/ilesha/Downloads/Swim Strokes'
#print(os.listdir(input_dir))
#for video_file in os.listdir(input_dir):
#   if video_file.endswith(('.mp4', '.mov')):
#       video_path = os.path.join(input_dir, video_file)
#       style=stroke_identifier(video_path)
#       stroke_style_count_dict = style.analyze_video(video_path)
#       stroke_style=style.final_stroke(stroke_style_count_dict)
#       if stroke_style=='Butterfly Stroke':
#           
#           counter= StrokeCounter(video_path, stroke_style)
#           counter.stroke_counter_forward()
#       else:
#           counter= StrokeCounter(video_path, stroke_style)
#           counter.stroke_counter_topview()
#           #counter.stroke_counter_topview(video_file)