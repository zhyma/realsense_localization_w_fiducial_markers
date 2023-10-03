 
import sys, copy, time, cv2
from cv2 import aruco

import numpy as np

def get_cv_version():
    v_str = [str(i) for i in str(cv2.__version__).split(".")]
    return int(v_str[0])*100 + int(v_str[1])

def get_marker_coords(k, distort, corners):
    ## from https://github.com/TemugeB/QR_code_orientation_OpenCV/blob/main/run_qr.py
    #Selected coordinate points for each corner of QR code.
    qr_edges = np.array([[0,0,0],
                         [0,1,0],
                         [1,1,0],
                         [1,0,0]], dtype = 'float32').reshape((4,1,3))

    #determine the orientation of QR code coordinate system with respect to camera coorindate system.
    ret, rvec, tvec = cv2.solvePnP(qr_edges, corners, np.reshape(k, (3,3)), np.array(distort))

    ## homogeneous transformation matrix
    ht = np.empty((4,4))
    ht[3, 3] = 1.0

    #Define unit xyz axes. These are then projected to camera view using the rotation matrix and translation vector.
    unitv_points = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype = 'float32').reshape((4,1,3))
    if ret:
        ht[:3, :3] = cv2.Rodrigues(rvec)[0]
        points, jac = cv2.projectPoints(unitv_points, rvec, tvec, np.reshape(k, (3,3)), np.array(distort))
        return points, ht

    #return empty arrays if rotation and translation values not found
    else: return [], []

def find_aruco(img, k, distort, id=1):
    # try:
    # aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)

    print(get_cv_version())
    
    if get_cv_version() <= 406:
        ## for opencv-contrib-python <= 4.6.0.66
        arucoDict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        arucoParams = aruco.DetectorParameters_create()
        corners, ids, rejected = aruco.detectMarkers(img, arucoDict, parameters=arucoParams)

    else:
        ## for opencv-python > =4.8 (and 4.7 maybe?)
        ## use the new API.
        arucoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        arucoParams = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(arucoDict, arucoParams)
        corners, ids, rejected = detector.detectMarkers(img)
    img_w_frame = copy.deepcopy(img)
    ht = None
    centroid = None
    if ids is not None:
        ids = ids.ravel().tolist()
        if id in ids:
            i = ids.index(id)
            ## Get the center as the coordinate frame center
            ## z-axis pointing out from the marker
            ## x-axis pointing up
            objPoints = np.array([[-1,1,0], [1,1,0], [1,-1,0], [-1,-1,0]], dtype = 'float32').reshape((4,1,3))
            valid, rvec, tvec= cv2.solvePnP(objPoints, corners[i], np.reshape(k, (3,3)), np.array(distort))
            if valid:
                ht = np.zeros((4,4))

                ht[3, 3] = 1.0
                ht[:3, :3] = cv2.Rodrigues(rvec)[0]

                centroid = (int(np.mean(corners[0][0][:,0])), int(np.mean(corners[0][0][:,1])))
                img_w_frame = cv2.circle(img_w_frame, centroid, radius = 10, color = (255, 255, 255), thickness=-1)

                aruco.drawDetectedMarkers(img_w_frame, corners)
                cv2.drawFrameAxes(img_w_frame, np.reshape(k, (3,3)), np.array(distort), rvec, tvec, 1)


    return img_w_frame, ht, centroid

def find_qr(img, k, distort, id=0):
    ## from https://github.com/TemugeB/QR_code_orientation_OpenCV/blob/main/run_qr.py
    #Selected coordinate points for each corner of QR code.

    ##IS NOT FINISHED!
    qr = cv2.QRCodeDetector()
    ret_qr, points = qr.detect(img)
    img_w_frame = copy.deepcopy(img)
    ht = None
    if ret_qr:
        axis_points, ht = get_marker_coords(k, distort, points)

        #BGR color format
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0,0,0)]

        #check axes points are projected to camera view.
        if len(axis_points) > 0:
            axis_points = axis_points.reshape((4,2))

            origin = (int(axis_points[0][0]),int(axis_points[0][1]) )

            for p, c in zip(axis_points[1:], colors[:3]):
                p = (int(p[0]), int(p[1]))

                #Sometimes qr detector will make a mistake and projected point will overflow integer value. We skip these cases. 
                if origin[0] > 5*img.shape[1] or origin[1] > 5*img.shape[1]:break
                if p[0] > 5*img.shape[1] or p[1] > 5*img.shape[1]:break

                cv2.line(img_w_frame, origin, p, c, 5)

    return img_w_frame, ht

if __name__ == '__main__':
    ...
