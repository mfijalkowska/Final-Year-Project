# -------------------------------------------
# 2021 Magdalena Fijalkowska, Liverpool, UK
# -------------------------------------------

import cv2 as cv


def detect_pose(frame, width, height, player_pose):

    # dictionary to store all body parts
    BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                  "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                  "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                  "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

    # list to store pairs of points that are connected to each other
    POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                  ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                  ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                  ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

    # dictionary to store crucial keyooints
    KEYPOINTS = {"RShoulder": [0, 0], "RElbow": [0, 0], "RWrist": [0, 0],
                 "LShoulder": [0, 0], "LElbow": [0, 0], "LWrist": [0, 0], "RHip": [0, 0], "RKnee": [0, 0],
                 "RAnkle": [0, 0], "LHip": [0, 0], "LKnee": [0, 0], "LAnkle": [0, 0]}

    # load inference graph
    net = cv.dnn.readNetFromTensorflow("pose/graph_opt.pb")

    # prepare the image
    #                                scaling    size        mean  convert BGR to RGB    no crop
    blob = cv.dnn.blobFromImage(frame, 1.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    # pass the blob as the input
    net.setInput(blob)
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1] as we only need the first 19 elements

    # throw assertion error if the num of body parts is not 19
    assert (len(BODY_PARTS) == out.shape[1])

    points = []   # list to store points detected
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        # We find global maximum (Note - only a single pose at the same time can be detected)
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (width * point[0]) / out.shape[3]
        y = (height * point[1]) / out.shape[2]

        # Add a point if its confidence is higher than threshold.
        if conf > 0.2:
            points.append((int(x), int(y)))
            # if body part is desired for angle calculations --> update values
            if str(list(BODY_PARTS)[i]) in KEYPOINTS:
                KEYPOINTS[list(BODY_PARTS)[i]] = [int(x), int(y)]   # append center coordinates of that particular body part
        else:
            points.append(None)  # if a body part not detected

    skeleton = []   # list
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        # if not satisfied throw an assertion error
        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            skeleton.append([points[idFrom], points[idTo]])

    return skeleton, player_pose, KEYPOINTS
