from os import stat
import numpy as np 
import face_recognition
import cv2 as cv 
import time
import pandas as pd
import pickle
import sys
#base_image = face_recognition.load_image_file('./base_images/photo-profile.jpg')
#base_image = cv.cvtColor(base_image, cv.COLOR_BGR2RGB)
#face_loc = face_recognition.face_locations(base_image)[0]
#base_image_encoding = face_recognition.face_encodings(base_image)[0]

def persons_recognition(test_image_copy,data_base,tm, state):
  ''' This function  suppose the input image to be RGB format.'''
  test_image_faces = face_recognition.face_locations(test_image_copy)
  test_image_encoding = face_recognition.face_encodings(test_image_copy)
  #print('I am invoked...')
  for i,encoding in enumerate(test_image_encoding) :
    #print("I am looped")
    if(len(data_base) > 0):
      historical_encoding = data_base.loc[:,"encoding"].values.tolist()
      matching_person = face_recognition.compare_faces(historical_encoding,encoding)
      matching_person = [o for o,val in enumerate(matching_person) if val == True]
      if (len(matching_person) > 0):
        id = data_base.loc[matching_person[0],"id"]
      else :
        state += 1
        id = state
    else:
      id = state
    new_data = pd.concat([data_base,pd.DataFrame({'id': id,'encoding': [encoding.tolist()], 'time': tm, 'make_purchase': 0})], ignore_index = True)
    new_data.to_csv('./historic.csv',index = False)
    return new_data, state
    #face = test_image_faces[i]
    #result = face_recognition.compare_faces([base_image_encoding],encoding)
    #distance = face_recognition.face_distance([base_image_encoding],encoding)
    #x1,y1,x2,y2 = face[3]*4, face[0]*4, face[1]*4,face[2]*4
    #cv.rectangle(test_image,(x1,y1),(x2,y2),(0,255,0),2)
    #cv.rectangle(test_image,(x1,y2-30),(x2,y2),(0,255,0),cv.FILLED)
    #text  = 'Taher' if result[0] else 'Unknown'
    #confidence = f"{(1 - int(distance[0])) * 100}%"
    #x_coor = x1 
    #y_coor = y2 - 5
    #cv.putText(test_image,text,(x_coor,y_coor),cv.FONT_HERSHEY_COMPLEX, 1, (255,255,255),1)
    #cv.putText(test_image,confidence,(x2 - int((x2-x1)/2), y_coor), cv.FONT_HERSHEY_COMPLEX, 1, (68, 88, 112), 1)
    #return test_image

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
    data_base = pd.DataFrame(columns=['id','encoding', 'time', 'make_purchase'])
  cap = cv.VideoCapture(0)
  print(state)
  while(True):
    try:
      _,frame = cap.read()
      frame_copy = cv.resize(frame,(0,0),None,0.25,0.25)
      entry_time = time.time()
      entry_time = time.ctime(entry_time)
      try :
        data_base, state= persons_recognition(frame_copy,data_base, entry_time, state)
      except TypeError:
        pass
      #cv.imshow('Watch_persons',image)
      #key = cv.waitKey(1) & 0xFF
      #if key == ord('q'):
      #break
    except KeyboardInterrupt:
      with open('./state_encoding','wb') as f:
        pickle.dump(state, f)
      sys.exit('programme stop monitoring the shop')

if __name__ == "__main__":
    record_and_detect()