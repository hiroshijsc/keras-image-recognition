from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import keras

model = VGG16()
image = load_img("madrid.jpg", target_size = (224, 224))
image = img_to_array(image)
image = np.expand_dims(image, axis = 0)
image = preprocess_input(image) #画像をアレイにしたときはこのコードを必ず使う

pred = model.predict(image) #predictions
print("Predicted:", decode_predictions(pred, top=3)[0])
np.argmax(pred[0])

# Grad-CAM アルゴリズム
specoutput=model.output[:, 668] # モデルの出力を取り出す
last_conv_layer = model.get_layer('block5_conv3') # 最後の畳み込み層を取り出す
grads = K.gradients(specoutput, last_conv_layer.output)[0] # 取り出したモデル出力の最終畳み込み層の勾配を求める（クラス分類の確率スコアへの影響の多さを測る）
                                                           #   → クラス判定に影響を与える画像箇所の微分係数を求める（画像箇所の変化率）
pooled_grads = K.mean(grads, axis=(0, 1, 2)) # 平面上での平均を取り出す
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]]) # 上の4つを関数化する
pooled_grads_value, conv_layer_output_value = iterate([image]) # 入力画像に対する最終畳込み層出力の値と勾配を求める
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i] # 勾配の平均をかけることで各クラスによる違いを明瞭にする
heatmap=np.mean(conv_layer_output_value, axis=-1)
# ヒートマップを作成
heatmap = np.maximum(heatmap, 0) # ReLUを通す
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

# 画像とヒートマップを重ねる
import cv2
img = cv2.imread("madrid.jpg")
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0])) #ヒートマップの幅と高さにリサイズ
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # 色を鮮やかにする
superimposed_img = heatmap * 0.4 + img # ヒートマップと画像を足すことで重ねている
cv2.imwrite('madrid_heatmap.jpg', superimposed_img)

# 画像をトリミングする
from PIL import Image
baseheight = 700
img = Image.open('madrid.jpg')
hpercent = (baseheight / float(img.size[1])) # 1はheight
width = int((float(img.size[0]) * float(hpercent))) #リサイズしつつ画像の比は保つ
img = img.resize((width, baseheight), Image.ANTIALIAS)
img.save('resizedimagemadrid.jpg')
h1=baseheight/2
w1=width/2
croppedIm = img.crop((0, 0, w1, h1)) # left, up, right, bottom
croppedIm.save('croppedmadrid1.jpg')
croppedIm = img.crop((0, h1, w1, baseheight)) # left, up, right, bottom
croppedIm.save('croppedmadrid2.jpg')
croppedIm = img.crop((w1, 0, width, h1)) # left, up, right, bottom
croppedIm.save('croppedmadrid3.jpg')
croppedIm = img.crop((w1, h1, width, baseheight)) # left, up, right, bottom
croppedIm.save('croppedmadrid4.jpg')

# stockholm
stockholm_image = load_img("stockholm.jpg", target_size = (224, 224))
stockholm_image = img_to_array(stockholm_image)
stockholm_image = np.expand_dims(stockholm_image, axis = 0)
stockholm_image = preprocess_input(stockholm_image) #画像をアレイにしたときはこのコードを必ず使う

stockholm_pred = model.predict(stockholm_image) #predictions
print("Predicted:", decode_predictions(stockholm_pred, top=3)[0])
np.argmax(stockholm_pred[0])

stockholm_pooled_grads_value, stockholm_conv_layer_output_value = iterate([stockholm_image]) # 入力画像に対する最終畳込み層出力の値と勾配を求める
for i in range(512):
    stockholm_conv_layer_output_value[:, :, i] *= stockholm_pooled_grads_value[i] # 勾配の平均をかけることで各クラスによる違いを明瞭にする
stockholm_heatmap=np.mean(stockholm_conv_layer_output_value, axis=-1)
# ヒートマップを作成
stockholm_heatmap = np.maximum(stockholm_heatmap, 0) # ReLUを通す
stockholm_heatmap /= np.max(stockholm_heatmap)
plt.matshow(stockholm_heatmap)
plt.show()

# 画像とヒートマップを重ねる
stockholm_img = cv2.imread("stockholm.jpg")
stockholm_heatmap = cv2.resize(stockholm_heatmap, (stockholm_img.shape[1], stockholm_img.shape[0])) #ヒートマップの幅と高さにリサイズ
stockholm_heatmap = np.uint8(255 * stockholm_heatmap)
stockholm_heatmap = cv2.applyColorMap(stockholm_heatmap, cv2.COLORMAP_JET) # 色を鮮やかにする
stockholm_superimposed_img = stockholm_heatmap * 0.4 + stockholm_img # ヒートマップと画像を足すことで重ねている
cv2.imwrite('stockholmheatmap.jpg', stockholm_superimposed_img)
