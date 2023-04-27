import os
import cv2
import numpy as np
def his_equal(gray):               # 直方圖等化
    h, w = gray.shape[:2]          # 取得圖片大小
    arr = np.zeros(256)            # 建立陣列
    for i in range(h):             # 取得照片內各個灰階值的數量
        for j in range(w):
            color = gray[i][j]
            arr[color] += 1
    probability = arr / (h * w)  # 每個灰階值出現機率
    cumulative = np.zeros(256)  # 累計機率
    for i in range(256):
        if i == 0:
            cumulative[i] = probability[i]
        else:
            cumulative[i] = cumulative[i - 1] + probability[i]

    for i in range(len(cumulative)):
        cumulative[i] = round(cumulative[i] * 255)  # 計算新的灰階值
    cumulative = cumulative.astype('uint8')        # 將資料轉成福符號數
    img2 = cumulative[gray]         # 將舊圖片帶入新的灰階值

    return img2
if __name__ == '__main__':

    test_dataset = os.listdir(r'./train_dataset')
    for photo in test_dataset:    # 讀取測試集中每張圖片
        i = 0
        for i in range(4501):
            img = cv2.imread(f'./train_dataset/{photo}/{photo+str(i)}.jpg')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 將圖片轉為灰階
            img = cv2.resize(img, (128, 128))   # 縮放成統一大小
            img = his_equal(img)                # 直方圖等化
            if(i<4000):
                cv2.imwrite(f'./his_train_dataset/{photo + str(i)}.jpg', img)
            else:
             cv2.imwrite(f'./his_test_dataset/{photo + str(i-4000)}.jpg', img)
   
        print(photo+" 完成...")