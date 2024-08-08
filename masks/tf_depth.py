from PIL import Image
import os

# 深度画像が保存されているフォルダのパス
input_folder = 'data/images/depth'
output_folder = 'data/images/de'

# 出力フォルダが存在しない場合、作成する
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 0.pngから35.pngまでのファイルを処理する
for i in range(36):
    filename = f'{i}.png'
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    # 画像を開く
    with Image.open(input_path) as img:
        # 画像を反転させる
        inverted_img = Image.eval(img, lambda x: 255 - x)
        # 反転した画像を保存する
        inverted_img.save(output_path)

print("すべての画像を反転しました。")
