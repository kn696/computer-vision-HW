import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 讀取影像
image_path = '.\\211365.jpg'  # 換成你的目標圖片檔名
img_bgr = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# 2. 設定你觀察出的分類規格 (HSV 門檻值)
# OpenCV HSV 範圍：H: 0-179, S: 0-255, V: 0-255
lower_green = np.array([20, 50, 0])    # H:20~60, S:50~255, V:不設限(0~255)
upper_green = np.array([60, 255, 255])

# 3. 根據規則建立布林遮罩 (Mask)
mask = cv2.inRange(img_hsv, lower_green, upper_green)

# 4. 形態學後處理 (Morphology) - 去除孤立微小雜訊與填補葉片小孔洞
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 開運算：消除微小雜點
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel) # 閉運算：填補葉片空隙

# 5. 套用遮罩提取植物區域
segmented_result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_clean)

# 6. 繪製最終對比圖並儲存（可用於報告）
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_result)
plt.title('Segmented Plant Area (H:20~60, S:50~255)')
plt.axis('off')

plt.tight_layout()
plt.savefig('segmentation_result.png', dpi=300)
plt.show()