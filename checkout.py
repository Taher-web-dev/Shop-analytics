import cv2 as cv 
import face_recognition
import time
import pandas as pd
import numpy as np 
def detect_customer_face():
  cap = cv.VideoCapture(0)
  loop = True
  while loop:
    _,frame = cap.read()
    frame_copy = np.copy(frame)
    resized_frame = cv.resize(frame,(0,0),None,0.25,0.25)
    faces = face_recognition.face_locations(resized_frame)
    key = cv.waitKey(1) & 0xFF
    for i, face in enumerate(faces):
      x1,y1,x2,y2 = face[3]*4 ,face[0] *4 ,face[1] *4 ,face[2] * 4
      cv.rectangle(frame_copy,(x1,y1),(x2,y2),(0,255,0))
      p1,z1,p2,z2 = x1,y2-50,x2,y2
      cv.rectangle(frame_copy,(p1,z1),(p2,z2),(0,255,0),cv.FILLED)
      cv.putText(frame_copy,'Does this the right customer ?',(x1+5,int((z1+z2)/2)),cv.FONT_HERSHEY_COMPLEX,1/3,(0,0,255),1)
      cv.imshow("client_picture",frame_copy)
      if key == ord('y'):
        face_client = frame[y1:y2,x1:x2,:]
        client_encoding = face_recognition.face_encodings(face_client)[0]
        data_base = pd.read_csv('./historic.csv')
        data_base["encoding"] = data_base["encoding"].apply(eval)
        encoding = data_base.loc[:,"encoding"].values.tolist()
        matching = face_recognition.compare_faces(encoding, client_encoding)
        ref = [r for r,val in enumerate(matching) if val==True][0]
        id = data_base.loc[ref,'id']
        ind = data_base.loc[data_base['id']==id,:].index
        right_ind = ind[-1]
        data_base.loc[right_ind,'make_purchase'] = 1
        data_base.to_csv('./historic.csv',index=False)
        loop = False
        break
      if key == ord('n'):
        print('it is not the true client')
        continue

      #data_base = pd.read_csv('./historic.csv')

    if key == ord('q'):
      break

if __name__ == "__main__":
  detect_customer_face()

      
