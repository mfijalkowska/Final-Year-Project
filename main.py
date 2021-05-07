# -------------------------------------------
# 2021 Magdalena Fijalkowska, Liverpool, UK
# -------------------------------------------

import sys
import numpy as np
from src.videoReader import getVideoStream


def calculate_results(shot_result):
    results = {'attempts': shot_result['attempts'], 'made': shot_result['made'], 'missed': shot_result['misses'],
               'elbow_made': 0.0, 'elbow_miss': 0.0, 'knee_made': 0.0, 'knee_miss': 0.0, 'release_made': 0.0,
               'release_miss': 0.0, 'avg_speed': 0.0}

    # check if not empty
    if shot_result['elbow_angle_made']:
        results['elbow_made'] = round(np.mean(shot_result['elbow_angle_made']), 2)
    if shot_result['elbow_angle_miss']:
        results['elbow_miss'] = round(np.mean(shot_result['elbow_angle_miss']), 2)
    if shot_result['knee_angle_made']:
        results['knee_made'] = round(np.mean(shot_result['knee_angle_made']), 2)
    if shot_result['knee_angle_miss']:
        results['knee_miss'] = round(np.mean(shot_result['knee_angle_miss']), 2)
    if shot_result['release_made']:
        results['release_made'] = round(np.mean(shot_result['release_made']), 2)
    if shot_result['release_miss']:
        results['release_miss'] = round(np.mean(shot_result['release_miss']), 2)
    if shot_result['avg_speed']:
        results['avg_speed'] = round(np.mean(shot_result['avg_speed']), 2)

    return results


def save_results(results):
    f = open("training_results.txt", "w")

    f.write("Total shots attempts :  " + str(results['attempts']))
    f.write("\nMade shots :  " + str(results['made']))
    f.write("\nMissed shots :  " + str(results['missed']))
    f.write("\nElbow Angle - made shots :  " + str(results['elbow_made']))
    f.write("\nElbow Angle - missed shots :  " + str(results['elbow_miss']))
    f.write("\nKnee Angle - made shots :  " + str(results['knee_made']))
    f.write("\nKnee Angle - missed shots :  " + str(results['knee_miss']))
    f.write("\nRelease angle - made shots :  " + str(results['release_made']))
    f.write("\nRelease angle - missed shots :  " + str(results['release_miss']))
    f.write("\nAverage release speed :  " + str(results['avg_speed']) + "m/s")

    f.close()


def upload_video(path):
    return getVideoStream(str(path))


def training_analyser(path):
    # get training results
    shot_result = upload_video(path)
    # calculate and format the results
    results = calculate_results(shot_result)
    # save training results from a video to a txt file
    save_results(results)


if __name__ == "__main__":
    training_analyser(sys.argv[1])
