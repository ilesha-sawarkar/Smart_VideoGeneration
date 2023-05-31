#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np
import math
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


#calculate the angle between three keypoints
def calculate_angle(a, b, c):

	a = (a.x, a.y)
	b = (b.x, b.y)
	c = (c.x, c.y)

	radians = np.arccos(
		np.dot(np.array(b) - np.array(a), np.array(c) - np.array(b))
		/ (np.linalg.norm(np.array(b) - np.array(a)) * np.linalg.norm(np.array(c) - np.array(b)))
	)
	return np.degrees(radians)



def calculate_entry_angle(landmarks):
	left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
	right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
	left_hand = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
	right_hand = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
	
	# Calculate the angle between the arm and  horizontal line
	entry_angle_left = math.degrees(math.atan2(left_hand.y - left_shoulder.y, left_hand.x - left_shoulder.x))
	entry_angle_right = math.degrees(math.atan2(right_hand.y - right_shoulder.y, right_hand.x - right_shoulder.x))
	
	return entry_angle_left, entry_angle_right


def correct_butterfly_stroke(video_path):

	mp_pose = mp.solutions.pose
	pose = mp_pose.Pose()
	cap = cv2.VideoCapture(video_path)
	stroke_count=0
	while cap.isOpened():
		ret, frame = cap.read()
		
		if not ret:
			break
		

		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		
		results = pose.process(image)

		if results.pose_landmarks is not None:

			
			landmarks = results.pose_landmarks.landmark
			entry_angle_left, entry_angle_right = calculate_entry_angle(landmarks)
				
			if entry_angle_left > 60 and entry_angle_right > 60:
				print("Bad butterfly stroke entry")
			else:
				print("Good butterfly stroke entry")

		mp_drawing.draw_landmarks(
			frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
		)
		
		# Display the frame
		cv2.imshow("Butterfly Stroke Correction", frame)
		
		# Break the loop on 'q' key press
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break
		

	cap.release()
	cv2.destroyAllWindows()
	#/Users/ilesha/Downloads/Swim Strokes/pexels-kindel-media-8685964-1080x1920-30fps.mp4
	#'/Users/ilesha/Downloads/Swim Strokes/pexels-kindel-media-8685964-1080x1920-30fps.mp4'

#/Users/ilesha/Downloads/pexels-tima-miroshnichenko-6012516-2160x3840-25fps.mp4
	
	
#bad butterfly '/Users/ilesha/Downloads/ButterflyStroke/badbutterfly2_za8sjNzX.mp4.crdownload'
video_path='/Users/ilesha/Downloads/pexels-tima-miroshnichenko-6012516-2160x3840-25fps.mp4'
correct_butterfly_stroke(video_path)
