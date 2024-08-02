import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

# 初始化全局變數
points = []
image = None


def select_points(event, x, y, flags, param):  # 區域選擇
    global points, image
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        img_copy = image.copy()  # 複製一張圖片避免紅點也被跟著圖片轉正

        for point in points:
            cv2.circle(img_copy, point, 5, (0, 0, 255), -1)  # 畫紅色點
        cv2.namedWindow("Select_Points", cv2.WINDOW_NORMAL)
        cv2.imshow("Select_Points", img_copy)  # 將紅色點新增到複製的圖片上
        if len(points) == 4:
            cv2.destroyWindow("Select_Points")  # 選了四個點後就關閉畫面


def perspective_transform(src, point_c):
    width = max(
        int(np.linalg.norm(point_c[0] - point_c[1])),
        int(np.linalg.norm(point_c[2] - point_c[3]))
    )  # 計算範數並取最大值來做為寬
    print(width)
    height = max(
        int(np.linalg.norm(point_c[0] - point_c[3])),
        int(np.linalg.norm(point_c[1] - point_c[2])),
    )  # 同上只是做為高
    #print(height)
    
    dstp = np.array([[0, 0], [width-1, 0], [width-1, height-1],
                    [0, height-1]], dtype="float32")
    #dstp = [[0,0],[400,0],[400,600],[0,600]]
    # 得到 perspective transform matrix
    M = cv2.getPerspectiveTransform(point_c, dstp)
    print(M)
    #print(f'dstp{dstp}')
    #print(f'point_c:{point_c}')
    # 執行 perspective transformation
    dst = cv2.warpPerspective(src, M, (width, height))
    return dst


def main():
    global image

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()  # 選擇圖片

    image = cv2.imread(file_path)  # 讀取圖片

    if image is None:  # 如果沒選
        # os.system('cls') # 清乾淨
        print(f"\033[31m載入圖片失敗\033[0m")
        return

    # 在選擇點之前創建一個圖像複製品來顯示紅點
    img_copy = image.copy()

    cv2.imshow("Select_Points", img_copy)  # 顯示讀取的圖片
    cv2.setMouseCallback("Select_Points", select_points)  # 選擇點
    cv2.waitKey(0)

    if len(points) == 4:
        print(points)
        point_c = np.array(points, dtype="float32")  # 座標矩陣
        dst = perspective_transform(image, point_c)
        cv2.namedWindow("Warped Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Warped Image", dst)  # 顯示轉正圖片
        cv2.imwrite("warped_image.png", dst)  # 儲存轉正圖片
        cv2.waitKey(0)
    else:
        print("還有點點沒選完")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
