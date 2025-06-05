
import cv2                            # 1. Import OpenCV for webcam and drawing functions
import mediapipe as mp                # 2. Import MediaPipe for hand tracking

# Set up MediaPipe’s hand detection and tracking model
media_pipe_hands = mp.solutions.hands
drawing_utils    = mp.solutions.drawing_utils
hand_detector    = media_pipe_hands.Hands(
    max_num_hands          = 1,      # only look for one hand
    min_detection_confidence = 0.7,   # ignore weak hand detections
    min_tracking_confidence  = 0.7    # ignore shaky landmark tracking
)

#These are the landmark indices for each fingertip in MediaPipe’s model
fingertip_landmark_indices = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

def count_raised_fingers(hand_landmarks):
    """
    Count how many fingers are raised on one hand.
    Input: hand_landmarrks = 21 points (x,y,z) for one hand
    Output: number of fingers held up (0 to 5)
    """
    finger_states = []               # list to hold 1 (up) or 0 (down) for each finger
 
    #Thumb logic: compare tip x vs joint x
    thumb_tip   = hand_landmarks.landmark[fingertip_landmark_indices[0]]
    thumb_joint = hand_landmarks.landmark[fingertip_landmark_indices[0] - 1]
    if thumb_tip.x < thumb_joint.x:
        finger_states.append(1)      # thumb is up
    else:
        finger_states.append(0)      # thumb is down

    #Other fingers logic: compare tip y vs PIP joint y
    for i in range(1, 5):
        fingertip = hand_landmarks.landmark[fingertip_landmark_indices[i]]
        pip_joint = hand_landmarks.landmark[fingertip_landmark_indices[i] - 2]
        if fingertip.y < pip_joint.y:
            finger_states.append(1)  # finger is up
        else:
            finger_states.append(0)  # finger is down

    return sum(finger_states)        # total fingers raised

# Open the default webcam (device 0)
camera_capture = cv2.VideoCapture(0)
if not camera_capture.isOpened():
    raise RuntimeError("Unable to access the webcam")  # stop if camera fails

#Main loop: run until user quits
while True:
    frame_received, original_frame = camera_capture.read()  # grab a frame
    if not frame_received:
        break                       # exit if frame not read

    mirrored_frame = cv2.flip(original_frame, 1)            # mirror image left-right
    rgb_frame      = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)
                                                           # convert BGR to RGB for MediaPipe

    detection_results = hand_detector.process(rgb_frame)    # detect hand landmarks

    # If at least one hand is found, process it
    if detection_results.multi_hand_landmarks:
        for hand_landmarks in detection_results.multi_hand_landmarks:
            # 10. Draw landmarks and connections on the mirrored frame
            drawing_utils.draw_landmarks(
                mirrored_frame,
                hand_landmarks,
                media_pipe_hands.HAND_CONNECTIONS
            )

            # 11. Count how many fingers are raised
            finger_count = count_raised_fingers(hand_landmarks)

            # 12. Decide what word to show based on finger count
            if finger_count == 0:
                display_text = "No fingers"
            elif finger_count == 2:
                display_text = "2 fingers"
            elif finger_count == 5:
                display_text = "5 Fingers "
            else:
                display_text = str(finger_count) + " FINGERS"

            # 13. Overlay the chosen text at position (30,100)
            cv2.putText(
                mirrored_frame,
                display_text,
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,                # font size
                (0, 255, 0),      # green color
                4                 # thickness
            )

    # 14. Show the processed frame in a window titled "Sign Language Demo"
    cv2.imshow("Sign Language Demo", mirrored_frame)

    # 15. Wait 1ms for keypress; if 'q' is pressed, quit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 16. Release the webcam and close all OpenCV windows
camera_capture.release()
cv2.destroyAllWindows()
