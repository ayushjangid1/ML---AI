# # create database for features of face

# import cv2
# import face_recognition as fr 
# import pandas as pd
# file_name = "face_data.csv"

# # live capturing
# vid = cv2.VideoCapture(0)
# fd = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# try:
#     face_db = pd.read_csv(file_name,index_col=0, sep='\t')
#     data = {
#         'name':face_db['name'].values.tolist(),
#         'encoding':face_db['encoding'].values.tolist(),
#     }
# except Exception as e:
#     print(e)
#     data = {'name':[], 'encoding':[]}

# names = data['name']
# enc = data['encoding']
# framelimit = 20
# frameCount = 0
# name = input("Enter your name: ")
# while True:
#     flag, img  = vid.read()
#     if flag:
#         faces = fd.detectMultiScale(
#             cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),
#             scaleFactor = 1.1,
#             minNeighbors = 5,
#             minSize = (50,50)
#             )
#         if len(faces) == 1:
#             x,y,w,h = faces[0]
#             img_face = img[y:y+h,x:x+w,:].copy()
#             img_face = cv2.resize(img_face, (400,400),interpolation=cv2.INTER_CUBIC)
#             face_encoding = fr.face_encodings(img_face)
#             if len(face_encoding) == 1:
#                 enc.append(face_encoding[0].tolist())
#                 names.append(name)
#                 frameCount += 1
#                 print(frameCount)
#                 cv2.putText(
#                     img, str(frameCount),(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255),8)
#                 if frameCount == framelimit:
#                     data = {'name':names,'encoding':enc}
#                     pd.DataFrame(data).to_csv(file_name)
#                     break
#         for x1,y1,w,h in faces:
#             # img_cropped = img[y1: y1+h , x1:x1+w,:]

#             cv2.rectangle(img, pt1=(x1,y1),pt2=(x1+w,y1+h),color=(0,0,255), thickness=4)    
#         cv2.imshow('preview',img)
#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             break
# cv2.destroyAllWindows()
# vid.release()

#register faces
import cv2
import face_recognition as fr 
import pandas as pd

fd=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
vid=cv2.VideoCapture(0)
name=input('Enter Your name')
frameLimit=20
frameCount=0
names=[]
enc=[]
while True:
    flag,img=vid.read()
    if flag:

        #processing
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces=fd.detectMultiScale(img_gray,scaleFactor=1.1,minNeighbors=5,minSize=(50,50))
        if len(faces)==1:
            x,y,w,h=faces[0]
            img_face=img[y:y+h,x:x+w,:].copy()
            img_face=cv2.resize(img_face,(400,400),cv2.INTER_CUBIC)
            face_encoding=fr.face_encodings(img_face)
            if len(face_encoding)==1:
                enc.append(face_encoding[0].tolist())
                names.append(name)
                frameCount +=1
                cv2.putText(img,str(frameCount),(30,30),cv2.FONT_HERSHEY_COMPLEX,1.5,(0,0,255),5)
                if frameCount == frameLimit:
                    try:
                        old_data=pd.read_csv('face_data.csv',index_col=0,sep='|')
                    except Exception as e:
                        print(e)
                    else:
                        enc_old=old_data['encoding'].values.tolist()#values :- convrtt thr numpy array into list
                        names_old=old_data['names'].values.tolist()
                        enc=enc_old+enc
                        names=names_old+names
                    data={'names':names,'encoding':enc} # create dict
                    pd.DataFrame(data).to_csv('face_data.csv',sep='|')
                    break
           # print(face_encoding)

        for x,y,w,h in faces:
            cv2.rectangle(img,pt1=(x,y),pt2=(x+w,y+h),color=(0,0,255),thickness=2)

        cv2.imshow('Preview',img)
        key=cv2.waitKey(1)
        if key==ord('q'):
            break
    else:
        print('No frames')
        break
    #sleep(0.1)

cv2.destroyAllWindows()
cv2.waitKey(1)
vid.release()
