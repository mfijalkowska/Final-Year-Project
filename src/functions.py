# -------------------------------------------
# 2021 Magdalena Fijalkowska, Liverpool, UK
# -------------------------------------------

import cv2
import numpy as np
from .pose_estimation import detect_pose
from .detections import object_detection


def calculate_distance(a, b):
    return ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** (1/2)


def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return round(np.degrees(angle), 2)


def getAngles(KEYPOINTS):
    # get right elbow angle key points
    RShoulderX, RShoulderY = KEYPOINTS.get("RShoulder")
    RElbowX, RElbowY = KEYPOINTS.get("RElbow")
    RWristX, RWristY = KEYPOINTS.get("RWrist")

    # get left elbow angle key points
    # LShoulderX, LShoulderY = KEYPOINTS.get("LShoulder")
    # LElbowX, LElbowY = KEYPOINTS.get("LElbow")
    # LWristX, LWristY = KEYPOINTS.get("LWrist")

    # get right knee angle key points
    RHipX, RHipY = KEYPOINTS.get("RHip")
    RKneeX, RKneeY = KEYPOINTS.get("RKnee")
    RAnkleX, RAnkleY = KEYPOINTS.get("RAnkle")

    # get left knee angle key points
    LHipX, LHipY = KEYPOINTS.get("LHip")
    LKneeX, LKneeY = KEYPOINTS.get("LKnee")
    LAnkleX, LAnkleY = KEYPOINTS.get("LAnkle")

    RelbowAngle = calculate_angle(np.array([RShoulderX, RShoulderY]), np.array([RElbowX, RElbowY]), np.array([RWristX, RWristY]))
    # LelbowAngle = calculate_angle(np.array([LShoulderX, LShoulderY]), np.array([LElbowX, LElbowY]), np.array([LWristX, LWristY]))
    RkneeAngle = calculate_angle(np.array([RHipX, RHipY]), np.array([RKneeX, RKneeY]), np.array([RAnkleX, RAnkleY]))
    LkneeAngle = calculate_angle(np.array([LHipX, LHipY]), np.array([LKneeX, LKneeY]), np.array([LAnkleX, LAnkleY]))


    elbowAngle = RelbowAngle
    kneeAngle = np.mean([RkneeAngle, LkneeAngle])
    return elbowAngle, kneeAngle


def plot_results(img, skeleton, indexes, boxes, confidences, classes, class_ids, shot_result, trace):

    font = cv2.FONT_HERSHEY_PLAIN
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            if label == 'ball':
                cv2.rectangle(img, (x, y), (x + w, y + h), (204, 0, 204), 2)
                cv2.putText(img, label + " " + confidence, (x + w, y - 10), font, 3, (204, 0, 204), 3)
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (225, 120, 40), 2)
                cv2.putText(img, label + " " + confidence, (x + w, y - 10), font, 3, (225, 0, 0), 3)

    idFrom = 0
    idTo = 1

    for i in skeleton:
        # if points[i][idFrom] and points[i][idTo]:
        cv2.line(img, i[idFrom], i[idTo], (0, 255, 0), 2)
        cv2.ellipse(img, i[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
        cv2.ellipse(img, i[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    for ball in trace['balls']:
        # get ball center coord
        xball = int(np.mean([ball[0], ball[0] + ball[2]]))
        yball = int(np.mean([ball[1], ball[1] + ball[3]]))
        if yball < trace['hoop_height'] != 0:
            cv2.circle(img, (xball, yball), 10, (0, 0, 255), 3)
        else:
            cv2.circle(img, (xball, yball), 10, (255, 0, 0), 3)
    print(trace['frames'])
    if trace['frames'] > 0:
        if shot_result['judgement'] == "SCORE":
            cv2.putText(img, shot_result['judgement'], (trace['judgement_coords'][0] + 50, trace['judgement_coords'][1] - 15), font, 5, (0, 255, 0), 5)
        else:
            cv2.putText(img, shot_result['judgement'], (trace['judgement_coords'][0] + 50, trace['judgement_coords'][1] - 15), font, 5, (0, 0, 255), 5)
        trace['frames'] -= 1
    return img


def frame_detection(frame, width, height, prev_detection, player_pose, shot_result, trace):

    # getting pose detections
    try:
        skeleton, player_pose, KEYPOINTS = detect_pose(frame, width, height, player_pose)
    except:
        print("Error with pose estimation")

    # getting YOLO - ball & hoop detections
    try:
        indexes, boxes, confidences, classes, class_ids, prev_detection, KEYPOINTS, trace = object_detection(frame, width, height, prev_detection, KEYPOINTS, trace)
    except:
        print("Error with YOLO")

    # calculate knee and arm angles
    try:
        elbowAngle, kneeAngle = getAngles(KEYPOINTS)
    except:
        elbowAngle = 0
        kneeAngle = 0

    # check if shooting
    # if both ball and hoop detected
    if prev_detection['ball&hoop_detected'] and len(prev_detection['ball']) > 3:
        # get ball points
        print('GETTING BALL AND HOOP POINTS')
        xmin = prev_detection['ball'][-1][0]  # x coordinates of upper left corner
        ymin = prev_detection['ball'][-1][1]  # y coordinates of upper left corner
        w = prev_detection['ball'][-1][2]     # width of the ball
        h = prev_detection['ball'][-1][3]     # height of the ball
        xmax = xmin + w                       # x coordinates of bottom right corner
        ymax = ymin + h                       # y coordinates of bottom right corner
        x_center = int(np.mean([xmin, xmax]))   # x center coordinates
        y_center = int(np.mean([ymin, ymax]))   # y center coordinates

        # hoop height
        hoop_height = prev_detection['hoop'][-1][1]  # y coordinate of upper left corner
        hoop_xmin = prev_detection['hoop'][-1][0]    # xmin
        hoop_xmax = prev_detection['hoop'][-1][0] + prev_detection['hoop'][-1][2]  # xmax = xmin + width

        trace['hoop_height'] = hoop_height
        # check if ball is above the hoop
        if y_center < hoop_height and not player_pose['ball_in_air']:
            # calculate the release angle and speed
            xball1 = int(np.mean([prev_detection['ball'][-2][0], prev_detection['ball'][-2][0] + prev_detection['ball'][-2][2]]))  # x coord of the 2nd last ball
            yball1 = int(np.mean([prev_detection['ball'][-2][1], prev_detection['ball'][-2][1] + prev_detection['ball'][-2][3]]))  # y coord of the 2nd last ball
            xball2 = int(np.mean([prev_detection['ball'][-3][0], prev_detection['ball'][-3][0] + prev_detection['ball'][-3][2]]))
            yball2 = int(np.mean([prev_detection['ball'][-3][1], prev_detection['ball'][-3][1] + prev_detection['ball'][-3][3]]))

            # calculate the speed
            time_diff = prev_detection['ball_time'][-2] - prev_detection['ball_time'][-3]
            seconds = time_diff.microseconds/1000000
            distance = calculate_distance(prev_detection['ball'][-2], prev_detection['ball'][-3])
            meters = distance/w * 0.25          # 0.25m --> average ball diameter
            speed = meters/seconds
            shot_result['avg_speed'].append(speed)

            # calculate the release angle
            shot_result['release_angle'] = calculate_angle(np.array([xball1, yball1]), np.array([xball2, yball2]), np.array([xball1, yball2]))

            # mark ball in the air as True
            player_pose['ball_in_air'] = True

        elif player_pose['ball_in_air'] and y_center > hoop_height:
            # print('ball just went below the hoop')
            # mark ball in the air as False
            player_pose['ball_in_air'] = False
            release_angle = shot_result['release_angle']
            # SHOT
            shot_result['attempts'] += 1
            trace['frames'] = 10
            trace['judgement_coords'] = [x_center, y_center]
            if hoop_xmax > x_center > hoop_xmin:
                # SHOT
                shot_result['made'] += 1
                shot_result['judgement'] = 'SCORE'
                shot_result['release_made'].append(release_angle)
                if elbowAngle != 0:
                    shot_result['elbow_angle_made'].append(elbowAngle)
                    print(elbowAngle)
                if kneeAngle != 0:
                    shot_result['knee_angle_made'].append(kneeAngle)
                    print(kneeAngle)
                print('made ')
            else:
                # MISS
                shot_result['misses'] += 1
                shot_result['judgement'] = 'MISS'
                shot_result['release_miss'].append(release_angle)
                if elbowAngle != 0:
                    shot_result['elbow_angle_miss'].append(elbowAngle)
                if kneeAngle != 0:
                    shot_result['knee_angle_miss'].append(kneeAngle)
                print('miss')


    # plot on a frame
    my_frame = plot_results(frame, skeleton, indexes, boxes, confidences, classes, class_ids, shot_result, trace)

    return my_frame, shot_result
