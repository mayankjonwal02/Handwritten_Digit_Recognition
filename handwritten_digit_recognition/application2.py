import cv2
import numpy as np
import tensorflow as tf

# load the trained model
model = tf.keras.models.load_model("handwritten.model")

# initialize the camera of the mobile device
cap = cv2.VideoCapture("http://192.168.198.198:8080/video")

# loop through the frames captured by the camera
while True:
    # read the frame from the camera
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # if the frame is read correctly
    if ret:
        # flip the frame horizontally
        #frame = cv2.flip(frame, 1)

        # resize the frame to 28x28 pixels
        resized_image = cv2.resize(frame, (28, 28))

        # # convert the resized image to grayscale
        # gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # Convert image to grayscale
        gray_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # Convert grayscale image to black and white (binary) image
        (thresh, black_white_img) = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


        # invert the grayscale image
        inverted_image = cv2.bitwise_not(black_white_img)

        # normalize the inverted image pixel values to the range [0, 1]
        normalized_image = inverted_image / 255.0

        # reshape the normalized image to match the input shape of the model
        input_image = normalized_image.reshape((1, 28, 28, 1))

        # make a prediction using the model
        prediction = model.predict(input_image)

        # get the predicted digit label
        digit_label = np.argmax(prediction)

        # write the predicted digit label on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (10, 120)
        font_scale = 5
        color = (0, 0, 255)
        thickness = 4
        cv2.putText(frame, str(digit_label), org, font, font_scale, color, thickness)

        # display the frame
        cv2.imshow("Handwritten Digit Recognition", frame)

    # if there is an error reading the frame
    else:
        break

    # wait for a key press
    key = cv2.waitKey(1)

    # if the "q" key is pressed, stop the loop
    if key == ord("q"):
        break

# release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
