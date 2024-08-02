import numpy as np
import cv2
import glob

# 給cv2.cornerSubPix()用的終止條件
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

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)

        # 將數據轉為字串格式
        matrix_text = f"相機矩陣:\n{camera_matrix}\n" \
                      f"旋轉向量:\n{rvecs}\n" \
                      f"平移向亮:\n{tvecs}\n"

        # 在圖像上繪製矩陣和向量數據
        y0, dy = 50, 30
        for i, line in enumerate(matrix_text.split('\n')):
            y = y0 + i * dy
            cv2.putText(img, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # 保存圖片
        output_filename = f"{fname}_with_matrix.bmp"
        cv2.imwrite(output_filename, img)

cv2.destroyAllWindows()
