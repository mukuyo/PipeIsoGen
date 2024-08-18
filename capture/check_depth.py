# import cv2
# import numpy as np

# def analyze_depth_image(image_path):
#     # 画像を読み込む
#     image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

#     # 画像が正しく読み込まれたか確認
#     if image is None:
#         print(f"Error: Unable to load image at {image_path}")
#         return

#     # 画像の基本情報
#     height, width = image.shape[:2]
#     num_channels = 1 if len(image.shape) == 2 else image.shape[2]
#     dtype = image.dtype

#     print(f"Image dimensions: {width} x {height}")
#     print(f"Number of channels: {num_channels}")
#     print(f"Data type: {dtype}")

#     # チャンネルとデータタイプに基づいて追加の分析を行う
#     if dtype == np.uint16:
#         print("The image is in uint16 format.")
#         if num_channels == 1:
#             print("The image is likely in millimeters.")
#         else:
#             print("The image might not be in millimeters.")
#     elif dtype == np.uint8:
#         print("The image is in uint8 format.")
#         if num_channels == 1:
#             print("The image might be in millimeters, but it's typically not as detailed.")
#         else:
#             print("The image is likely a color image or has a different format.")
#     else:
#         print("The image format is not recognized.")

# if __name__ == "__main__":
#     # テストする画像パスを指定
#     image_path = 'SAM-6D/data/Example/depth.png'
#     analyze_depth_image(image_path)

import cv2
import numpy as np

# Depth画像のパスを指定
depth_image_path = 'SAM-6D/data/Example/depth.png'

# 画像をグレースケールで読み込み（Depth画像は通常グレースケール）
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

# 画像が読み込めたかチェック
if depth_image is None:
    print("画像の読み込みに失敗しました")
    exit()

# 画像の中央のピクセル位置を計算
height, width = depth_image.shape
center_y, center_x = height // 2, width // 2

# 中央ピクセルの距離を取得
distance = depth_image[center_y, center_x]

# 結果をプリント
print(f"中央ピクセルの距離: {distance} メートル")
