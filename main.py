import cv2
import os
import sys
import numpy
#0 is a primary source
haar_file = "C:/Users/amitt/Documents/openCV/Lesson-10/haarcascade_frontalface_default.xml"
# All the faces data will be
#  present this folder

datasets = "C:/Users/amitt/Documents/openCV/Lesson-10/data_sets"
#These are sub data sets of folder,
#for my faces I've used my name you can
#change the label here
sub_data = "Atharv"
print("Recognizing Face, please be in sufficient light")
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        path = os.path.join(datasets, subdir)
        for filename in os.listdir(path):
            file = path+'/'+filename
            label = id
            images.append(cv2.imread(file, 0))
            labels.append(int(label))
        id+=1

print(labels)
(width, height) = (130, 100)
(images, labels) = [numpy.array(lis) for lis in[images, labels]]
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, labels)
Faces = cv2.CascadeClassifier(haar_file)
myvideo = cv2.VideoCapture(0)
for i in range(30):
    return_value, image = myvideo.read()
    storegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = Faces.detectMultiScale(storegray, 1.4, 4)
    #print(face)
    for (x, y, w, h) in face:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 40, 132), 1)
        pic = storegray[y:y+h, x:x+w]
        pic_resize = cv2.resize(pic, (150, 100))
        prediction = recognizer.predict(pic_resize)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        if prediction[1] > 60:
            cv2.putText(image, '% s - %.0f' % (names[prediction[0]], prediction[1]), (x-10, y-10), cv2.FONT_ITALIC, 2, (104, 52, 78))
        else:
            cv2.putText(image, "Face not recognized!", (x-10, y-10), cv2.FONT_ITALIC, 2, (104, 52, 78))

    cv2.imshow("Face detection", image)
    key = cv2.waitKey(10)
    if key == 27:
        break




myvideo.release()
