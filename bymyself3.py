import cv2 as cv
import cv2
import numpy as np
import mediapipe as mp
import AiPhile
import time
from imutils.video import VideoStream
from imutils.video import FPS
# 初始化FaceMesh模型


mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
#========================
# variables
# distance from camera to object(face) measured
KNOWN_DISTANCE = 76.2  # centimeter
# width of face in the real world or Object Plane
KNOWN_WIDTH = 14.3  # centimeter

# Colors  >>> BGR Format(BLUE, GREEN, RED)

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 242)
GOLDEN = (32, 218, 165)
LIGHT_BLUE = (255, 9, 2)
PURPLE = (128, 0, 128)
CHOCOLATE = (30, 105, 210)
PINK = (147, 20, 255)
ORANGE = (0, 69, 255)

fonts = cv2.FONT_HERSHEY_COMPLEX
fonts2 = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
fonts3 = cv2.FONT_HERSHEY_COMPLEX_SMALL
fonts4 = cv2.FONT_HERSHEY_TRIPLEX
# Camera Object
cap = cv2.VideoCapture(0)  # Number According to your Camera
Distance_level = 0

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output21.mp4", fourcc, 30.0, (640, 480))

# face detector object
face_detector = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
# focal length finder function

# 定义其他必要变量和函数（如FocalLength、Distance_finder等）

def focal_length(measured_distance, real_width, width_in_rf_image):
    """
    This Function Calculate the Focal Length(distance between lens to CMOS sensor), it is simple constant we can find by using
    MEASURED_DISTACE, REAL_WIDTH(Actual width of object) and WIDTH_OF_OBJECT_IN_IMAGE
    :param1 Measure_Distance(int): It is distance measured from object to the Camera while Capturing Reference image

    :param2 Real_Width(int): It is Actual width of object, in real world (like My face width is = 14.3 centimeters)
    :param3 Width_In_Image(int): It is object width in the frame /image in our case in the reference image(found by Face detector)
    :retrun focal_length(Float):"""
    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value


# distance estimation function

def distance_finder(focal_length, real_face_width, face_width_in_frame):
    """
    This Function simply Estimates the distance between object and camera using arguments(focal_length, Actual_object_width, Object_width_in_the_image)
    :param1 focal_length(float): return by the focal_length_Finder function

    :param2 Real_Width(int): It is Actual width of object, in real world (like My face width is = 5.7 Inches)
    :param3 object_Width_Frame(int): width of object in the image(frame in our case, using Video feed)
    :return Distance(float) : distance Estimated
    """
    distance = (real_face_width * focal_length) / face_width_in_frame
    return distance

# face detector function
def face_data(image):
    """
    This function Detect the face
    :param Takes image as argument.
    :returns face_width in the pixels
    """

    face_width = 0
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, h, w) in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), WHITE, 1)
        face_width = w

    return face_width

# reading reference image from directory
ref_image = cv.imread("Ref_image.png")

ref_image_face_width = face_data(ref_image)
focal_length_found = focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, ref_image_face_width)
print(focal_length_found)
print(ref_image_face_width)

vs = VideoStream(src=0, usePiCamera=False, resolution=(640, 480), framerate=30).start()
fps = FPS().start()

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    # frame_counter = 0
    # starting_time = time.time()
    while True:
        # frame_counter += 1
        frame = vs.read()
        face_width_in_frame = face_data(frame)

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            # 进行眼睛和虹膜检测，你提供的代码段
            # ...
            # print(results.multi_face_landmarks[0].landmark)
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            # print(mesh_points.shape)
            # cv.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
            # cv.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            cv.circle(frame, center_left, int(l_radius), (255,0,255), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (255,0,255), 1, cv.LINE_AA)




            # 进行人脸距离检测，你提供的人脸距离检测代码

        if face_width_in_frame != 0:
            Distance = distance_finder(focal_length_found, KNOWN_WIDTH, face_width_in_frame)
            # Drwaing Text on the screen
            # cv.putText(
            #     frame, f"Distance = {round(Distance,2)} CM", (50, 50), fonts, 1, (WHITE), 2)
            AiPhile.textBGoutline(
                 frame,
                 f"Distance = {round(Distance, 2)} CM",
                 (30, 80),
                 scaling=0.5,
                 text_color=AiPhile.YELLOW,
                )

        # fps = frame_counter / (time.time() - starting_time)
        # AiPhile.textBGoutline(frame, f"FPS: {round(fps, 2)}", (30, 40), scaling=0.5)
        cv.imshow("frame", frame)
        if cv.waitKey(1) == ord("q"):
            break
        fps.update()

fps.stop()
print(fps.fps())
cv.destroyAllWindows()
vs.stop()

