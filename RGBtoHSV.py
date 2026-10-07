import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. 讀取 CSV 資料
csv_file = 'pixel_samples2.csv'
df = pd.read_csv(csv_file)

# 2. 將 RGB 轉為 OpenCV 的 HSV 格式
# 注意：OpenCV 的 RGB 輸入範圍為 [0, 255]
# 轉出來的 HSV 範圍：H ∈ [0, 179], S ∈ [0, 255], V ∈ [0, 255]
rgb_pixels = df[['R', 'G', 'B']].values.astype(np.uint8)
# 將 shape (N, 3) 轉為 OpenCV 需要的 (N, 1, 3) 圖像形狀
rgb_reshaped = rgb_pixels.reshape(-1, 1, 3)
hsv_reshaped = cv2.cvtColor(rgb_reshaped, cv2.COLOR_RGB2HSV)
hsv_pixels = hsv_reshaped.reshape(-1, 3)

# 將 H, S, V 分別存回 DataFrame 方便過濾
df['H'] = hsv_pixels[:, 0]
df['S'] = hsv_pixels[:, 1]
df['V'] = hsv_pixels[:, 2]

# 分離植物與背景點
plant_df = df[df['Label'] == 1]
bg_df = df[df['Label'] == 0]

# 3. 繪製 H-S 2D 散佈圖
plt.figure(figsize=(9, 7))

# 畫植物點（綠色圓點）
plt.scatter(plant_df['H'], plant_df['S'], c='green', alpha=0.75, 
            edgecolors='k', s=60, label=f'Plant (n={len(plant_df)})')

# 畫背景點（紅色十字）
plt.scatter(bg_df['H'], bg_df['S'], c='red', marker='x', alpha=0.75, 
            s=60, label=f'Background (n={len(bg_df)})')

# 4. 圖表細節設定（作業報告評分重點）
plt.title('2D Color Space Scatter Plot (HSV: H-S Plane)', fontsize=14, fontweight='bold')
plt.xlabel('Hue (H) [0 - 179]', fontsize=12)
plt.ylabel('Saturation (S) [0 - 255]', fontsize=12)
plt.xlim(0, 180)
plt.ylim(0, 256)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11)

# 可選：加註 OpenCV 的常見綠色區間標示線（供參考觀察）
plt.axvspan(35, 85, color='green', alpha=0.1, label='Typical Green Hue Zone')

plt.tight_layout()
plt.savefig('hs_scatter_plot.png', dpi=300) # 自動存成高品質圖檔放入報告
plt.show()