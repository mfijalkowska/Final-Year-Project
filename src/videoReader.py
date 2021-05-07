# -------------------------------------------
# 2021 Magdalena Fijalkowska, Liverpool, UK
# -------------------------------------------

import cv2
from .functions import frame_detection

# rescale the frame
def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def getVideoStream(vidPath):
    # objects to store detection status
    prev_detection = {      # up to 10
        'ball&hoop_detected': False,
        'ball': [],  # xmin, ymax, xmax, ymin
        'hoop': [],  # xmin, ymax, xmax, ymin
        'ball_time': [],
    }
    trace = {
        'balls': [],
        'hoop_height': 0,
        'frames': 0,
        'judgement_coords': [],
        'release_point': []
    }
    player_pose = {
        'ball_in_air': False,
        'elbow_angle': 370,
        'knee_angle': 370,
    }
    shot_result = {
        'judgement': "",
        'attempts': 0,
        'misses': 0,
        'made': 0,
        'elbow_angle_made': [],
        'knee_angle_made': [],
        'elbow_angle_miss': [],
        'knee_angle_miss': [],
        'release_angle': 0.0,
        'release_made': [],
        'release_miss': [],
        'avg_speed': []
    }

    cap = cv2.VideoCapture(vidPath)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv2.waitKey()
            break
        frame = rescale_frame(frame, 80)
        # extract height and width
        height, width, _ = frame.shape

        detection, shot_result = frame_detection(frame, width, height, prev_detection, player_pose, shot_result, trace)
        # print(shot_result)    # output current stats in console
        cv2.imshow('Basketball Training', detection)

    # When everything is finished, release the capture
    cap.release()
    cv2.destroyAllWindows()
    # print("Final shooting result:")
    # print(shot_result)
    return shot_result

