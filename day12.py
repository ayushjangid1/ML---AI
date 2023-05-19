import cv2
import matplotlib.pyplot as plt

import cv2
# for face recognition
fd = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# for smile detection
sd = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

#start the camera and start capturing
vid = cv2.VideoCapture(0)

seq = 0

captured = False
while not captured:

    # vid.read is capturing image
    flag, img = vid.read()
    if flag:
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        x1, y1, w, h = (100, 100 , 200, 300)

        faces = fd.detectMultiScale(
            img_gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (180,180)
            )

        for x1,y1,w,h in faces:
            face = img[y1: y1+h , x1:x1+w].copy()

            smiles = sd.detectMultiScale(face,1.1,5,minSize=(20,20))

            if len(smiles) == 1:
                xs,ys,ws,hs = smiles[0]
                # showing smile
                cv2.rectangle(img,pt1=(xs+x1,ys+y1),pt2=(xs+ws+x1,ys+hs+y1),color=(0,255,0),thickness=2)
                # saving image and changing captured = True
                seq = seq + 1
                if seq == 5:
                    captured = cv2.imwrite('face1.png',img)
                    break
            else:
                seq=0

            cv2.rectangle(img, pt1=(x1,y1),pt2=(x1+w,y1+h),color=(0,0,255), thickness=4)
        cv2.imshow('Preview',img)
        key = cv2.waitKey(1)
        if key == ord('x'):
            break
    else:
        break
    # sleep(0.1)
vid.release()
cv2.destroyAllWindows()