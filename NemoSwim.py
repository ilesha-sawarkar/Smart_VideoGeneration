from StrokeCounter import StrokeCounter as stroke_counter
from stroke_identification2 import stroke_identifiication as stroke_identifier
from Butterfly_Stroke_and_Freestyle_correction import stroke_correction as corrector
import os
import cv2
import mediapipe as mp
import math


print("HI  !")
print('What is your choice. Please Enter:')
print('1: Counting Strokes and Identification of Strokes')
print('2: Correct your Butterfly Stroke or Freestyle Stroke')
choice= input('Enter 1 or 2: ')

if choice=='1':
	print('You have selected Counting and Identification of Strokes')
	input_dir = '/Users/ilesha/Downloads/Swim Strokes'
	print(os.listdir(input_dir))
	for video_file in os.listdir(input_dir):
		if video_file.endswith(('.mp4', '.mov')):
			video_path = os.path.join(input_dir, video_file)
			style=stroke_identifier(video_path)
			stroke_style_count_dict = style.analyze_video(video_path)
			stroke_style=style.final_stroke(stroke_style_count_dict)
			if stroke_style=='Butterfly Stroke':
				
				counter= stroke_counter(video_path, stroke_style)
				counter.stroke_counter_forward()
			else:
				counter= stroke_counter(video_path, stroke_style)
				counter.stroke_counter_topview()
				#counter.stroke_counter_topview(video_file)
			
else:
	print('You have selected Counting and Identification of Strokes')
	choice_stroke=int(input('Enter 1 or 2 for Butterfly or Freestyle Stroke correction'))
	if choice_stroke==1:
		input_dir = '/Users/ilesha/Downloads/ButterflyStroke/'
		print(os.listdir(input_dir))
		for video_file in os.listdir(input_dir):
			if video_file.endswith(('.mp4', '.mov')):
				stroke_style_count_dict = {}
				
				video_path = os.path.join(input_dir, video_file)
				correct= corrector(video_path)
				print(f"Video: {video_file}")
				correct.correct_butterfly_stroke(video_path)
				print("-----------------------------------")
	else:
		input_dir = '/Users/ilesha/Downloads/FrontCrawl Stroke/'
		print(os.listdir(input_dir))
		for video_file in os.listdir(input_dir):
			if video_file.endswith(('.mp4', '.mov')):
				stroke_style_count_dict = {}
				video_path = os.path.join(input_dir, video_file)
				correct= corrector(video_path)
				print(f"Video: {video_file}")
				correct.correct_front_crawl_stroke(video_path)
				print("-----------------------------------")