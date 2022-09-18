import numpy as np
import cv2
import os



#KNN CODE

def dist(x1, x2):
    return np.sqrt(sum((x1 - x2) ** 2))


def KNN(train,test, k=5):
    vals = []
    m = train.shape[0]

    for i in range(m):
        ix=train[i,:-1]
        iy=train[i,-1]
        d=dist(test,ix)
        vals.append((d,iy))
        #d = dist(point, train[i])
        #vals.append((d, test[i]))


    vals = sorted(vals)
    vals = vals[:k]

    vals = np.array(vals)
    # print(vals)

    new_vals = np.unique(vals[:, 1], return_counts=True)
    # print(new_vals)

    index = np.argmax(new_vals[1])
    pred = new_vals[0][index]

    return pred

##########################

cap = cv2.VideoCapture(0)
face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
skip = 0
face_data = []
dataset_path= './face_recognition/'
labels =[]
class_id=0
names = {}

#DATA PREPARATION

for fx in os.listdir(dataset_path):
    names[class_id]= fx[:-4]
    data_item = np.load(dataset_path+fx)
    face_data.append(data_item)
    target = class_id*np.ones((data_item.shape[0],))
    class_id+=1
    labels.append(target)
face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape(-1,1)

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels),axis= 1)
print(trainset.shape)

#TESTING

while True:
    ret, frame = cap.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret:
        faces = face_cascade.detectMultiScale(frame,1.3,5)

        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            offset = 10
            face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
            face_section = cv2.resize(face_section,(100,100))

            #PREDICTION
            out = KNN(trainset,face_section.flatten())

            #DISPLAY NAME
            name_info = names[int(out)]
            cv2.putText(frame,name_info,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),3)

        cv2.imshow('Faces', frame)

    key = cv2.waitKey(1)
    if key & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
