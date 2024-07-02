# Adapted from code in 
# https://datahacker.rs/009-how-to-detect-facial-landmarks-using-dlib-and-opencv/
# https://pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/

# Necessary imports (for preprocessing images)
import cv2
import dlib
import numpy as np
from google.colab.patches import cv2_imshow

# The intended photo format
GOAL_WIDTH = 224
GOAL_HEIGHT = 224
GOAL_LEFT_EYE = (0.27, 0.27)
GOAL_RIGHT_EYE = (1-GOAL_LEFT_EYE[0], GOAL_LEFT_EYE[1])

# Given a filepath, returns an image object
def read_image_from_file(filepath):
    return cv2.imread(filepath)

# Given an image, displays it (designed to work in Google Colab)
def show_image(img):
    cv2_imshow(img)

# Given an image filepath, displays it (designed to work in Google Colab)
def show_image_from_file(filepath):
    img = cv2.imread(filepath)
    cv2_imshow(img)

# Given an image, returns it as a keras input
def preprocess_image(image):
	# swap color channels, resize the input image, and add a batch
	# dimension
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))
	image = np.expand_dims(image, axis=0)
	# return the preprocessed image
	return image

# Given the filepath of any image (.jpg, .jpeg, or .png), locates and 
# centers the face, then crops the image to the desired shape
# Returns a cv2 image object
def crop_image(filepath, goal_width=GOAL_WIDTH, goal_height=GOAL_HEIGHT,
               goal_left_eye=GOAL_LEFT_EYE, goal_right_eye=GOAL_RIGHT_EYE):
    img = cv2.imread(filepath)
    # convert to grayscale
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize dlib's face detector
    detector = dlib.get_frontal_face_detector()
    # Detecting faces in the grayscale image
    faces = detector(gray)
    face = faces[0]

    # Extract specific coordinates of the main face (x1,x2,y1,y2)
    x1=face.left()
    y1=face.top()
    x2=face.right()
    y2=face.bottom()

    # Initialize dlib's shape predictor
    p = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(p)

    # Get the shape using the predictor
    landmarks=predictor(gray, face)

    # find center of both eyes
    left_eye_x = int((landmarks.part(37).x + landmarks.part(38).x +
                landmarks.part(40).x + landmarks.part(41).x) / 4)
    left_eye_y = int((landmarks.part(37).y + landmarks.part(38).y +
                landmarks.part(40).y + landmarks.part(41).y) / 4)
    right_eye_x = int((landmarks.part(43).x + landmarks.part(44).x +
                landmarks.part(46).x + landmarks.part(47).x) / 4)
    right_eye_y = int((landmarks.part(43).y + landmarks.part(44).y +
                landmarks.part(46).y + landmarks.part(47).y) / 4)

    # Calculate the angle of the eyes (tanx = dY/dX) ((0,0) is at top left)
    dY = -(right_eye_y - left_eye_y)
    dX = right_eye_x - left_eye_x
    angle = (np.degrees(np.arctan(dY/dX)))

    # Find the center between both eyes
    eye_center = ((right_eye_x + left_eye_x) // 2, (right_eye_y + left_eye_y) // 2)

    # ratio between current photo scale and goal scale
    curr_eye_dist = np.sqrt((dX**2) + (dY**2))
    goal_eye_dist = goal_width * (goal_right_eye[0]-goal_left_eye[0])
    scale = goal_eye_dist / curr_eye_dist

    # Create the rotation and scaling matrix (fixed at the center of the eyes)
    M = cv2.getRotationMatrix2D(eye_center, -(angle), scale)

    # Where to locate eye_center in the image
    tX = goal_width * 0.5
    tY = goal_height * goal_left_eye[1]

    # Add a translation component to the rotate+scale matrix
    M[0, 2] += (tX - eye_center[0])
    M[1, 2] += (tY - eye_center[1])

    # Perform the transformation
    # More info on the warp method https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html
    (w,h) = (goal_width, goal_height)

    return cv2.warpAffine(img, M, (w, h),flags=cv2.INTER_CUBIC)

