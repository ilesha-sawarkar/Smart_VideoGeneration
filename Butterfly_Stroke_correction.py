#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np
import math
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Function to calculate the angle between three keypoints
def calculate_angle(a, b, c):
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



def calculate_entry_angle(landmarks):
	# Extract relevant landmarks for analysis
	left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
	right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
	left_hand = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
	right_hand = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
	
	# Calculate the angle between the arm and the horizontal line
	entry_angle_left = math.degrees(math.atan2(left_hand.y - left_shoulder.y, left_hand.x - left_shoulder.x))
	entry_angle_right = math.degrees(math.atan2(right_hand.y - right_shoulder.y, right_hand.x - right_shoulder.x))
	
	return entry_angle_left, entry_angle_right

# Main function for stroke correction
def correct_butterfly_stroke(video_path):
	# Initialize Mediapipe Pose and VideoCapture
	mp_pose = mp.solutions.pose
	pose = mp_pose.Pose()
	cap = cv2.VideoCapture(video_path)
	stroke_count=0
	while cap.isOpened():
		ret, frame = cap.read()
		
		if not ret:
			break
		
		# Convert the BGR frame to RGB
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		
		# Process the frame with Mediapipe
		results = pose.process(image)
		
		# Check if any keypoints are detected
		if results.pose_landmarks is not None:
			# Retrieve the keypoints for the butterfly stroke
			#keypoints = results.pose_landmarks.landmark
			
			landmarks = results.pose_landmarks.landmark
			entry_angle_left, entry_angle_right = calculate_entry_angle(landmarks)
				
			if entry_angle_left > 60 and entry_angle_right > 60:
				print("Bad butterfly stroke entry")
			else:
				print("Good butterfly stroke entry")
				
			# Specify the indices for relevant keypoints
			#right_shoulder = keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
			#right_elbow = keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
			#right_wrist = keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value]
			
			# Calculate the angle between the shoulder, elbow, and wrist
				#angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
			
			# Perform stroke correction based on the angle
				#if angle > 10:
				# Angle is less than 120 degrees, indicating incorrect technique
				#print("Correct your butterfly stroke technique!")
				# Angle is 120 degrees or greater, indicating correct technique
				
				#print("Your butterfly stroke technique is good!")
				
		# Render the keypoints on the frame
		mp_drawing.draw_landmarks(
			frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
		)
		
		# Display the frame
		cv2.imshow("Butterfly Stroke Correction", frame)
		
		# Break the loop on 'q' key press
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break
		
	# Release the VideoCapture and close all windows
	cap.release()
	cv2.destroyAllWindows()
	#/Users/ilesha/Downloads/Swim Strokes/pexels-kindel-media-8685964-1080x1920-30fps.mp4
	#'/Users/ilesha/Downloads/Swim Strokes/pexels-kindel-media-8685964-1080x1920-30fps.mp4'

#/Users/ilesha/Downloads/pexels-tima-miroshnichenko-6012516-2160x3840-25fps.mp4
	
	
#bad butterfly '/Users/ilesha/Downloads/ButterflyStroke/badbutterfly2_za8sjNzX.mp4.crdownload'
video_path='/Users/ilesha/Downloads/pexels-tima-miroshnichenko-6012516-2160x3840-25fps.mp4'
correct_butterfly_stroke(video_path)
