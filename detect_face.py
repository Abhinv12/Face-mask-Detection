import cv2

def detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = classifier.detectMultiScale(gray,
                                        scaleFactor = 1.2)
    
    '''for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)'''
    return faces
