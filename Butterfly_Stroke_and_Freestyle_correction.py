import cv2
import mediapipe as mp
import numpy as np
import os
import math


class stroke_correction():
	def __init__(self, video_path="video.mp4"):
		self.video_path=video_path
		self.mp_drawing = mp.solutions.drawing_utils
		self.mp_pose = mp.solutions.pose
	def calculate_angle(a, b, c):
	
		a = (a.x, a.y)
		b = (b.x, b.y)
		c = (c.x, c.y)
	
		radians = np.arccos(
			np.dot(np.array(b) - np.array(a), np.array(c) - np.array(b))
			/ (np.linalg.norm(np.array(b) - np.array(a)) * np.linalg.norm(np.array(c) - np.array(b)))
		)
		return np.degrees(radians)
	
	
	def calculate_catch_angle(self,landmarks):
		# Extract relevant landmarks for analysis
		left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
		right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
		left_elbow = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
		right_elbow = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
		left_hand = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
		right_hand = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
		
		# Calculate the catch angle for left and right arm
		catch_angle_left = math.degrees(math.atan2(left_hand.y - left_elbow.y, left_hand.x - left_elbow.x) -
										math.atan2(left_elbow.y - left_shoulder.y, left_elbow.x - left_shoulder.x))
		catch_angle_right = math.degrees(math.atan2(right_hand.y - right_elbow.y, right_hand.x - right_elbow.x) -
										math.atan2(right_elbow.y - right_shoulder.y, right_elbow.x - right_shoulder.x))
		
		return catch_angle_left, catch_angle_right
	
	def correct_front_crawl_stroke(self, video_path):
		self.mp_pose = mp.solutions.pose
		pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
		cap = cv2.VideoCapture(video_path)
		good_total=0
		bad_total=0
		average=0
		total=0
		good_stroke_text='Good Front Crawl Stroke Percentage = ' + str(average)+ ' %'
		while cap.isOpened():
			success, frame = cap.read()
			if not success:
				break
			
			# Preprocess the frame
			
			# Convert the frame to RGB format
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			
			# Process the frame with MediaPipe Pose
			results = pose.process(frame_rgb)
			
			if results.pose_landmarks:
				landmarks = results.pose_landmarks.landmark
				catch_angle_left, catch_angle_right = self.calculate_catch_angle(landmarks)
				
				#good_stroke_text='Good Stroke Percentage % = ' + str(average)
				if catch_angle_left < 120 and catch_angle_right < 120:
					print("Good front crawl angle")
					cv2.rectangle(frame, (10, 10), (200, 200), (0, 255, 0), -1)  # Draw green rectangle for good stroke
					good_total+=1
				else:
					print("Bad front crawl angle. The arm angle has to be greater than 120")
					cv2.rectangle(frame, (10, 10), (200, 200), (0, 0, 255), -1) 
					bad_total+=1
				if (good_total and bad_total!=0):
					total=good_total+bad_total
					average=good_total/total
					average=round(average,4)*100
			good_stroke_text='Good Front Crawl Stroke Percentage = ' + str(average)+ ' %'
			text_size = cv2.getTextSize(good_stroke_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
			text_x = frame.shape[1] - 20 - text_size[0]  # Calculate the x position of the text based on the frame width
			text_y = 50 
			cv2.putText(frame, good_stroke_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) # white
			cv2.imshow("Front Crawl Stroke Correction", frame)
			
			# Break the loop on 'q' key press
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
		
		cap.release()
		cv2.destroyAllWindows()
	
	def calculate_entry_angle(self,landmarks):
		left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
		right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
		left_hand = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
		right_hand = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
		
		# Calculate the angle between the arm and  horizontal line
		entry_angle_left = math.degrees(math.atan2(left_hand.y - left_shoulder.y, left_hand.x - left_shoulder.x))
		entry_angle_right = math.degrees(math.atan2(right_hand.y - right_shoulder.y, right_hand.x - right_shoulder.x))
		
		return entry_angle_left, entry_angle_right
	
	
	def correct_butterfly_stroke(self, video_path):
	
		self.mp_pose = mp.solutions.pose
		pose = self.mp_pose.Pose()
		cap = cv2.VideoCapture(video_path)
		
		good_total=0
		bad_total=0
		average=0
		total=0
		good_stroke_text='Good Butterfly Stroke Percentage = ' + str(average)+ ' %'
		
		while cap.isOpened():
			ret, frame = cap.read()
			
			if not ret:
				break
			
	
			image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			
			results = pose.process(image)
	
			if results.pose_landmarks is not None:
	
				
				landmarks = results.pose_landmarks.landmark
				entry_angle_left, entry_angle_right = self.calculate_entry_angle(landmarks)
					
				if entry_angle_left > 60 and entry_angle_right > 60:
					print("Bad butterfly stroke entry angle greater than 60! ")
					cv2.rectangle(frame, (10, 10), (200, 200), (0, 0, 255), -1) 
					bad_total+=1
				else:
					print("Good butterfly stroke entry")
					cv2.rectangle(frame, (10, 10), (200, 200), (0, 255, 0), -1)
					good_total+=1
				if (good_total and bad_total!=0):
					total=good_total+bad_total
					average=good_total/total
					average=round(average,4)*100
					
				good_stroke_text='Good Butterfly Stroke Percentage = ' + str(average)+ ' %'
			self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
			
			
			
			
			text_size = cv2.getTextSize(good_stroke_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
			text_x = frame.shape[1] - 20 - text_size[0]  # Calculate the x position of the text based on the frame width
			text_y = 50 
			cv2.putText(frame, good_stroke_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) # white
			cv2.imshow("Butterfly Stroke Correction", frame)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
			
	
		cap.release()
		cv2.destroyAllWindows()
		
		


		

		