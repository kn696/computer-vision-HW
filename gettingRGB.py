import cv2
import pandas as pd
import numpy as np
import os

# ==================== 設定參數 ====================
IMAGE_PATH = '.\\309972.jpg'  # 換成你想標記的圖片路徑
OUTPUT_CSV = 'pixel_samples2.csv'
# ==================================================

# 儲存採樣數據: [R, G, B, Label, X, Y]
samples = []
# 歷史點位紀錄（用於繪圖與撤銷功能）
click_history = [] 

def mouse_callback(event, x, y, flags, param):
    global samples, click_history, img_display, img_clean

    # 左鍵點擊：採樣植物 (Label = 1, 標記綠色)
    if event == cv2.EVENT_LBUTTONDOWN:
        b, g, r = img_clean[y, x]
        samples.append({'R': r, 'G': g, 'B': b, 'Label': 1, 'X': x, 'Y': y})
        click_history.append({'pos': (x, y), 'color': (0, 255, 0), 'label': 1})
        print(f"[植物 Plant] (X={x}, Y={y}) -> R:{r}, G:{g}, B:{b}")

    # 右鍵點擊：採樣背景/非植物 (Label = 0, 標記紅色)
    elif event == cv2.EVENT_RBUTTONDOWN:
        b, g, r = img_clean[y, x]
        samples.append({'R': r, 'G': g, 'B': b, 'Label': 0, 'X': x, 'Y': y})
        click_history.append({'pos': (x, y), 'color': (0, 0, 255), 'label': 0})
        print(f"[背景 Background] (X={x}, Y={y}) -> R:{r}, G:{g}, B:{b}")

    # 更新畫面
    update_display()

def update_display():
    global img_display, img_clean
    img_display = img_clean.copy()
    
    plant_count = sum(1 for s in samples if s['Label'] == 1)
    bg_count = sum(1 for s in samples if s['Label'] == 0)

    # 在圖上記錄點擊位置
    for item in click_history:
        cv2.circle(img_display, item['pos'], 3, item['color'], -1)

    # 顯示即時計數統計
    text_plant = f"Plant (Left Click): {plant_count}/50"
    text_bg = f"Background (Right Click): {bg_count}/50"
    
    cv2.putText(img_display, text_plant, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img_display, text_bg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img_display, "Press 'S' to Save | 'Z' to Undo | 'Q' to Quit", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Pixel Sampler", img_display)

def main():
    global img_clean, img_display

    if not os.path.exists(IMAGE_PATH):
        print(f"錯誤：找不到圖片檔案 '{IMAGE_PATH}'，請確認檔名與路徑。")
        return

    # 讀取原始圖片
    img_clean = cv2.imread(IMAGE_PATH)
    
    # 若圖片尺寸過大（如 1164x1600），可適當縮放以利螢幕顯示（非必要，可依需求調整）
    height, width = img_clean.shape[:2]
    max_dim = 1000
    if max(height, width) > max_dim:
        scale = max_dim / float(max(height, width))
        img_clean = cv2.resize(img_clean, (int(width * scale), int(height * scale)))

    img_display = img_clean.copy()

    cv2.namedWindow("Pixel Sampler")
    cv2.setMouseCallback("Pixel Sampler", mouse_callback)

    update_display()

    print("================ 操作說明 ================")
    print(" - 滑鼠【左鍵】：標記「植物」 (Plant)")
    print(" - 滑鼠【右鍵】：標記「非植物/背景」 (Background)")
    print(" - 按鍵  'z'  ：撤銷上一點 (Undo)")
    print(" - 按鍵  's'  ：儲存目前採樣點至 CSV")
    print(" - 按鍵  'q'  ：結束程式")
    print("==========================================")

    while True:
        key = cv2.waitKey(1) & 0xFF

        # 按 'z' 撤銷上一筆
        if key == ord('z') or key == ord('Z'):
            if samples:
                removed = samples.pop()
                click_history.pop()
                print(f"已撤銷最後一點: {removed}")
                update_display()
            else:
                print("目前沒有可撤銷的點。")

        # 按 's' 儲存數據
        elif key == ord('s') or key == ord('S'):
            if len(samples) == 0:
                print("尚未採樣任何點，無法儲存。")
            else:
                df = pd.DataFrame(samples)
                df.to_csv(OUTPUT_CSV, index=False)
                print(f"\n[成功] 已將 {len(df)} 筆數據儲存至 '{OUTPUT_CSV}'！")

        # 按 'q' 或 ESC 退出
        elif key == ord('q') or key == ord('Q') or key == 27:
            # 退出前詢問是否儲存
            if samples and not os.path.exists(OUTPUT_CSV):
                df = pd.DataFrame(samples)
                df.to_csv(OUTPUT_CSV, index=False)
                print(f"\n[自動儲存] 已將 {len(df)} 筆數據儲存至 '{OUTPUT_CSV}'。")
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()