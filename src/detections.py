# -------------------------------------------
# 2021 Magdalena Fijalkowska, Liverpool, UK
# -------------------------------------------

import datetime
import cv2
import numpy as np


def object_detection(frame, width, height, prev_detection, KEYPOINTS, trace):
    # load yolo conf & weights
    net = cv2.dnn.readNet('yolo/yolov4-obj_final.weights', 'yolo/yolov4-obj.cfg')

    # print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    # print(cv2.cuda.getCudaEnabledDeviceCount())

    # load classes file and put them into a list
    with open("yolo/classes.txt", "r") as f:
        classes = f.read().splitlines()

    # prepare the image             Normalize the image pixel values (divide by 255)
    #                                scaling    size        mean  convert BGR to RGB    no crop
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    # print(blob.shape)
    # pass the blob as the input
    net.setInput(blob)
    # Returns names of layers with unconnected outputs
    output_layers_names = net.getUnconnectedOutLayersNames()
    # obtain outputs at the output layer
    layerOutputs = net.forward(output_layers_names)
    # extract bounding boxes
    boxes = []
    # store confidences
    confidences = []
    # store classes IDs
    class_ids = []

    for output in layerOutputs:  # extract all the info from the layers output
        for detection in output:   # to extract the info from each of the output(detection) contains (center_x, center_y, w, h, confidence, classes)
            # store all the 2 classes predictions starting from the 6th element to the end
            scores = detection[5:]
            # identify classes that have the highest score in this scores vector
            class_id = np.argmax(scores)
            #pass this element to identify the maximum element from this class --> which is confidence/probability
            confidence = scores[class_id]
            if confidence > 0.4:
                center_x = int(detection[0]*width)   # we need to scale it back cos we used blob before
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                # get the position of the upper left corner
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)
    # reset
    prev_detection['ball&hoop_detected'] = False
    # let's take the ball and hoop coordinates
    if len(indexes) > 0:
        for i in indexes.flatten():
            label = str(classes[class_ids[i]])
            x, y, w, h = boxes[i]

            # if both hoop and ball detected --> update values
            if len(indexes) == 2:
                prev_detection['ball&hoop_detected'] = True
                if len(prev_detection[label]) >= 10:
                    prev_detection['ball'].pop(0)
                    prev_detection['hoop'].pop(0)
                    prev_detection['ball_time'].pop(0)

                prev_detection[label].append([x, y, w, h])
                if label == 'ball':
                    prev_detection['ball_time'].append(datetime.datetime.now())

            if label == 'ball':
                if len(trace['balls']) > 5:
                    trace['balls'].pop(0)
                trace['balls'].append([x, y, w, h])

    return indexes, boxes, confidences, classes, class_ids, prev_detection, KEYPOINTS, trace
