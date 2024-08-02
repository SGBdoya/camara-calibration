import numpy as np
import cv2
import glob
import pandas as pd

# 給cv2.conrnerSubPix()用的終止條件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 棋盤格子數量-1
objp = np.zeros((7*7, 3), np.float32)  # 三維
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)  # 二維

# 儲存點的list
objpoints = []  # 3d 現實中的空間
imgpoints = []  # 2d 平面上的點

images = glob.glob('*.bmp')


for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 找棋盤角落
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

    # 如果有就加上點
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # 畫點
        cv2.drawChessboardCorners(img, (7, 7), corners2, ret)

        # 顯示
        # cv2.imshow('img', img)
        # cv2.imwrite(f'{fname}_point.bmp', img)
        cv2.waitKey(0)
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)

        # 顯示結果
        print(f"{fname}")
        print(f'測試ret是什麼: {ret}')
        print("相機矩陣(camera matrix):\n", camera_matrix)
        print("畸變係數(distortion coefficients):\n", dist_coeffs)
        print("旋轉向量(rotation vectors):\n", rvecs)
        print("平移向量(translation vectors):\n", tvecs)


cv2.destroyAllWindows()
