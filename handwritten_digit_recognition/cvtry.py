import cv2
import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model("handwritten.model")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame1=frame #cv2.flip(frame,1)
    h,w,c = frame.shape
    print(h,w)
    # Resize the image using cv2.resize()
    resized_image = cv2.resize(frame, (28, 28))
    # Reshape the resized image to match the input shape of the Keras model
    resized_image = np.expand_dims(resized_image[:, :, 0], axis=-1)
    resized_image = np.expand_dims(resized_image, axis=0)
    prediction = model.predict(resized_image)
    print(np.argmax(prediction))

    # write text on the image
    text = str(np.argmax(prediction))
    org = (50, 50) # top-left corner of the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 0, 255) # white color
    thickness = 2
    cv2.putText(frame1, text, org, font, fontScale, color, thickness)
    
    cv2.imshow("hello",frame1)
    cv2.waitKey(1)