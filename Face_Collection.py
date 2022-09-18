import cv2
import numpy as np
cap = cv2.VideoCapture(0)
face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
skip = 0
face_data = []
dataset_path= './face_recognition/'
filename= input("Enter the name of person: ")
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret==False:
        continue

    faces = face_cascade.detectMultiScale(gray,1.3,5)
    faces=sorted(faces,key=lambda f:f[2]*f[3])

    for(x,y,w,h) in faces[-1:]:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))

        skip+=1
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))


        cv2.imshow('Frame', frame)
        cv2.imshow('Face Section',face_section)

    key = cv2.waitKey(1)
    if key & 0xff == ord('q'):
        break

face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

np.save(dataset_path+filename+'.npy',face_data)
print("Data Successfully saved at " , dataset_path ,"as", filename +'.npy' )

cap.release()
cv2.destroyAllWindows()