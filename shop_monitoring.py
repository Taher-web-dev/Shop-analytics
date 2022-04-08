#! env/bin/python3.8
from os import stat
from turtle import pos
import numpy as np 
import face_recognition
import cv2 as cv 
import time
import pandas as pd
import pickle
import sys
from standard_positions import check_the_right_position

def persons_recognition(test_image_copy,data_base,tm, state, flow_persons):
  ''' This function  suppose the input image to be RGB format.'''
  test_image_faces = face_recognition.face_locations(test_image_copy)
  test_image_encoding = face_recognition.face_encodings(test_image_copy)
  new_data = data_base
  for i,encoding in enumerate(test_image_encoding) :
    try:
      face = test_image_faces[i]
    except :
      continue
    check_frame = face_recognition.compare_faces(flow_persons,encoding, tolerance=0.6)
    check_true = [t for t,val in enumerate(check_frame) if val == True]
    if len(check_true) > 0:
      break
    id = 0
    name = 'unknown'
    if(len(new_data) > 0):
      historical_encoding = new_data.loc[:,"encoding"].values.tolist()
      matching_person = face_recognition.compare_faces(historical_encoding,encoding,tolerance=0.6)
      matching_person = [o for o,val in enumerate(matching_person) if val == True]
      if (len(matching_person) > 0):
        id = new_data.loc[matching_person[0],"id"]
        names = new_data.loc[new_data["id"] == id,"name"].values.tolist()
        names = [ val for val in names if val!='']
        if len(names)>0 :
          name = names[0]
      else :
        state += 1
        id = state
    else:
      id = state
    new_data = pd.concat([new_data,pd.DataFrame({'id': id,'encoding': [encoding.tolist()], 'time': tm, 'make_purchase': 0,'name':name})], ignore_index = True)
    flow_persons = test_image_encoding
  new_data.to_csv('./historic.csv',index = False) 
  return new_data, state, flow_persons
   

def record_and_detect():
  state = 0
  with open('./state_encoding', 'rb') as f :
    try:
      state = pickle.load(f)
    except:
      pass

  try:
    data_base = pd.read_csv('./historic.csv')
    data_base["encoding"] = data_base["encoding"].apply(eval)
  except:
    data_base = pd.DataFrame(columns=['id','encoding', 'time', 'make_purchase','name'])
  cap = cv.VideoCapture(0)
  flow_persons = []
  while(True):
    try:
      _,frame = cap.read()
      frame_copy = cv.resize(frame,(0,0),None,0.25,0.25)
      frame_copy = cv.cvtColor(frame_copy,cv.COLOR_BGR2RGB)
      entry_time = time.time()
      entry_time = time.ctime(entry_time)
      try :
        data_base, state, flow_persons= persons_recognition(frame_copy,data_base, entry_time, state,flow_persons)
      except TypeError:
        pass
    except KeyboardInterrupt:
      with open('./state_encoding','wb') as f:
        pickle.dump(state, f)
      sys.exit('programme stop monitoring the shop')

if __name__ == "__main__":
    record_and_detect()