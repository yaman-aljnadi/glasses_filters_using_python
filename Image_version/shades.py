from tensorflow.keras.models import load_model
import cv2 
import numpy as np 
import time 

my_model = load_model("glasses_model.h5")

face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")

filters = "glasses/sunglasses_4.png"
image = "images/6.jpg"

def ratio_func_hight(galsses_original_shape, glasses_hight_shape, original_image_shape):
    percentege = (glasses_hight_shape/galsses_original_shape)*100
    print(percentege)
    hight = (original_image_shape/100)*percentege
    print(int(hight))
    return int(hight)


def ratio_func_width(galsses_original_shape, glasses_width_shape, original_image_shape):
    percentege = (glasses_width_shape/galsses_original_shape)*100
    print(percentege)
    width = (original_image_shape/100)*percentege
    print(int(width))
    return int(width)

cap = cv2.VideoCapture(0)
running = False
timer = time.time()

while running:
    ret, frame = cap.read()

    x_shape = frame.shape 
    cv2.putText(frame, f"focus_on_the_middle {int (timer-time.time())}", (x_shape[0]-40,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2) 
    cv2.imshow("video", frame)
    cv2.waitKey(1)
    if int(time.time() - timer) == 5:
        cv2.imwrite("last_frame/frame.jpg", frame)
        running = False 

frame = cv2.imread(image, cv2.IMREAD_UNCHANGED)
frame2 = np.copy(frame)
frame3 = np.copy(frame)

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(gray, 1.25, 6)

for (x,y,w,h) in faces:
    gray_face = gray[y:y+h, x:x+w]
    color_face = frame[y:y+h, x:x+w]
    color_face_copy = np.copy(color_face)

    gray_normalized = gray_face/255

    (x_original, y_original) = gray_face.shape  

    face_resized = cv2.resize(gray_normalized, (96,96))
    face_resized_copy = face_resized.copy()
    face_resized = face_resized.reshape(1,96,96,1)

    keypoints = my_model.predict(face_resized)

    keypoints = keypoints * 48 + 48

    face_resized_color = cv2.resize(color_face, (96,96))
    face_resized_color2 = np.copy(face_resized_color)

    points = []
    for i, co , in enumerate(keypoints [0][0::2]):
        points.append((co, keypoints[0][1::2][i]))

    sunglasses = cv2.imread(filters, cv2.IMREAD_UNCHANGED)
    sunglasses_width = int((points[7][0]-points[9][0])*1.1)
    print(sunglasses_width)
    sunglasses_hight = int((points[10][1]-points[8][1])/1.1)
    print(sunglasses_width)
    sunglasses_resized = cv2.resize(sunglasses, (sunglasses_width, sunglasses_hight))
    transparent_region = sunglasses_resized[:,:,:3] != 0
    face_resized_color[int(points[9][1]):int(points[9][1]) + sunglasses_hight, int(points[9][0]): int(points[9][0]) + sunglasses_width,:][transparent_region] = sunglasses_resized[:,:,:3][transparent_region]

    frame[y:y+h, x:x+w] = cv2.resize(face_resized_color, (x_original, y_original))


    test_frame = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    sunglasses_test = cv2.imread(filters)
    sunglasses_test = cv2.flip(sunglasses_test, 0)
    row_test, column_test, alpha = test_frame[y:y+h, x:x+w].shape
    width = ratio_func_width(96, sunglasses_width+5, row_test)
    hight = ratio_func_hight(96, sunglasses_hight-5,column_test)
    sunglasses_test = cv2.resize(sunglasses_test, (width, hight))

    print(sunglasses_test.shape)
    gw,gh,gc = sunglasses_test.shape

    for i in range(0,gw):
        for j in range(0, gh):
            if sunglasses_test[i,j][2] != 0:
                try:
                    color_face_copy[y-i-350, x+j-240] = sunglasses_test[i,j]
                except:
                    pass

    test_frame[y:y+h, x:x+w] = cv2.resize(color_face_copy, (x_original, y_original))

    for keypoints in points[:]:
        cv2.circle(face_resized_color2, keypoints, 1,(0,255,0), 1)

    frame2[y:y+h, x:x+w] = cv2.resize(face_resized_color2, (x_original,y_original))

    frame = cv2.resize(frame, (800,800))
    frame2 = cv2.resize(frame2, (800,800))
    frame3 = cv2.resize(frame3, (800,800))
    actual_frame = cv2.resize(test_frame, (800,800))

    cv2.imshow("filter", frame)
    cv2.imshow("keys", frame2)
    cv2.imshow("dst", actual_frame)
    cv2.imshow("AAA", frame3)


cv2.waitKey(0) & 0xff == ord("q")
cv2.destroyAllWindows()





    












