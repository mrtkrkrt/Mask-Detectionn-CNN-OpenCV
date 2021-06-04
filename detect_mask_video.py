import cv2
import numpy as np
import pickle
import tensorflow as tf

cap = cv2.VideoCapture(0)

cap.set(3,480)
cap.set(4,480)

classes = ["No Mask", "Mask"]
colors = [(0,255,0), (255,0,0)]

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

model = tf.keras.models.load_model('model.model')

while True:
    
    isTrue, img = cap.read()
    faces = face_cascade.detectMultiScale(img, 1.1, 3)

    for (x, y, w, h) in faces:
        frame = img[y:y+h, x:x+w, :]
        frame = cv2.resize(img, (32, 32))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = tf.keras.preprocessing.image.img_to_array(frame)
        frame = frame / 255.
        frame = tf.expand_dims(frame, axis=0)

        result = np.argmax(model.predict(frame)[0])
        print(result)
        text = classes[result]
        color = colors[result]
        
        cv2.rectangle(img, pt1 = (x, y), pt2 = (x + w, y + h), color = color, thickness = 2)
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        
    cv2.imshow("Mask Detection",img)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()