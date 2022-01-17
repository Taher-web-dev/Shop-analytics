from logging import critical
import cv2 as cv 
import numpy as np 
import dlib
import face_recognition
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 

pretained_model = './shape_predictor_68_face_landmarks.dat'
face_pose_predictor = dlib.shape_predictor(pretained_model)

def shapes_to_np(shape):
  np_shapes = np.zeros((68,2),dtype='int')
  for i in range(68):
    np_shapes[i] = (shape.part(i).x,shape.part(i).y)
  return np_shapes

critical_face_landmarks = {
  "nose" : (27,35),
  "right_eye": (36, 42),
  "left_eye": (42, 48)
}
face_detector = dlib.get_frontal_face_detector()

def check_the_right_position(frame):
  """This function suppose that the introduced image has only one face"""
  face = face_detector(frame)[0]
  test = True
  shape = face_pose_predictor(frame,face)
  coords = shapes_to_np(shape)
  rg_nose = critical_face_landmarks["nose"]
  max_nose = 0
  min_nose = float('inf')
  for m in range(rg_nose[0], rg_nose[1]):
    if coords[m,0] < min_nose :
      min_nose = coords[m,0]
    if coords[m,0] > max_nose :
      max_nose = coords[m,0]
  avg_nose = (min_nose + max_nose) / 2 
  rg_left_eye = critical_face_landmarks['left_eye']
  max_left = 0
  for m in range(rg_left_eye[0],rg_left_eye[1]):
    if coords[m,0] > max_left:
      max_left = coords[m,0]
  min_right = float('inf')
  rg_right_eye = critical_face_landmarks["right_eye"]
  for m in range(rg_right_eye[0],rg_right_eye[1]):
    if(coords[m,0] < min_right):
      min_right = coords[m,0]
  min_distance = min(abs(min_right - avg_nose), abs(max_left - avg_nose))
  max_distance = max(abs(min_right - avg_nose), abs(max_left - avg_nose))
  if (max_distance > (min_distance + 0.25 * min_distance)):
      #cv.putText(frame,'not good position',(150,50),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0),1)
      #cv.imshow("frame", frame)
    return False
    #cv.imshow("frame", frame)
  return True



