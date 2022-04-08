import cv2 as cv
import face_recognition
from face_recognition.api import face_encodings
import numpy as np
import pandas as pd
import time
def customer_informations() :
  cap = cv.VideoCapture(0)
  t_0 = time.time()
  while True:
    data_base = pd.read_csv('./historic.csv')
    data_base["encoding"] = data_base["encoding"].apply(eval)
    _,frame = cap.read()
    resized_frame = cv.resize(frame,(0,0),None,0.25,0.25)
    resized_frame = cv.cvtColor(resized_frame,cv.COLOR_BGR2RGB)
    copy_frame = np.copy(frame)
    faces = face_recognition.face_locations(resized_frame)
    encoding_cl = face_recognition.face_encodings(resized_frame)
    key = cv.waitKey(1) & 0xFF
    for i, face in enumerate(faces):
      x1, y1, x2, y2 = face[3] * 4, face[0] * 4, face[1] * 4, face[2] * 4
      p1, q1, p2, q2 = x1, y2, x2, y2 + 120
      n1, m1, n2, m2 = x1, y1 -80, x2, y1
      cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,0))
      cv.rectangle(frame,(p1,q1),(p2,q2),(0,255,0),cv.FILLED)
      cv.rectangle(frame,(n1,m1),(n2,m2),(0,255,0),cv.FILLED)
      face_image = copy_frame[y1:y2,x1:x2,:]
      client_encoding = encoding_cl[i]
      encoding = data_base.loc[:,"encoding"].values.tolist()
      matching = face_recognition.compare_faces(encoding,client_encoding)
      ind_matching = [i for (i,v) in enumerate(matching) if v==True]
      if (len(ind_matching) > 1):
        ind = ind_matching[0]
        id = data_base.loc[ind,'id']
        ext_data_base = data_base.loc[data_base["id"] == id, :]
        ext_ind = ext_data_base.index
        if(len(ext_ind)> 1):
          right_data_base = data_base.loc[ext_ind[:-1],:]
        else:
          right_data_base = data_base.loc[ext_ind,:]
        names = right_data_base.loc[:,'name'].values.tolist()
        name = names[0]
        nb_visite = len(right_data_base)
        nb_purchasing = right_data_base.loc[:,'make_purchase'].sum()
        chance = round((nb_purchasing/nb_visite),2)
        observation = 'Old'
        last_visite = right_data_base.loc[ext_ind[-2],'time']
        last_visite = pd.to_datetime(last_visite)
        text = f'{observation} Visitor'
        cv.putText(frame,text,(p1+5,q1+20),cv.FONT_HERSHEY_COMPLEX,0.3,(255,255,255),1)
        text = f'Visite Number: {nb_visite}'
        cv.putText(frame,text,(p1+5,q1+40),cv.FONT_HERSHEY_COMPLEX,0.3,(255,255,255),1)
        text = f'Number of transactions: {nb_purchasing}'
        cv.putText(frame,text,(p1+5,q1+60),cv.FONT_HERSHEY_COMPLEX,0.3,(255,255,255),1)
        text = f'Last Visite: {last_visite}'
        cv.putText(frame,text,(p1+5,q1+80),cv.FONT_HERSHEY_COMPLEX,0.3,(255,255,255),1)
        cor1 = n1 + int((n2-n1)/4)
        cor2 = m1 + int((m2 - m1)/2)
        cv.putText(frame,str(name),(cor1, cor2),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
      else: 
        text = 'New visitor'
        cv.putText(frame,text,(p1 + int((p2-p1)/4),q1 + int((q2-q1)/2)),cv.FONT_HERSHEY_COMPLEX,1/2,(255,255,255),1)
      cv.imshow('current_frame',frame)
    if key == ord('q'):
      break

if __name__ == "__main__":
  customer_informations()