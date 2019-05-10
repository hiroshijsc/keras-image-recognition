from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import keras
import cv2

model = VGG16()
pil_image = load_img("croppedmadrid4.jpg", target_size = (224, 224))
pil_image = img_to_array(pil_image)
pil_image = np.expand_dims(pil_image, axis = 0)
pil_image = preprocess_input(pil_image)
pil_pred = model.predict(pil_image) #predictions
print("Predicted:", decode_predictions(pil_pred, top=3)[0])
np.argmax(pil_pred[0])

specoutput=model.output[:, 668] # モデルの出力を取り出す
last_conv_layer = model.get_layer('block5_conv3') # 最後の畳み込み層を取り出す
grads = K.gradients(specoutput, last_conv_layer.output)[0] # 取り出したモデル出力の最終畳み込み層の勾配を求める（クラス分類の確率スコアへの影響の多さを測る）
                                                           #   → クラス判定に影響を与える画像箇所の微分係数を求める（画像箇所の変化率）
pooled_grads = K.mean(grads, axis=(0, 1, 2)) # 平面上での平均を取り出す
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]]) # 上の4つを関数化する
pil_pooled_grads_value, pil_conv_layer_output_value = iterate([pil_image]) # 入力画像に対する最終畳込み層出力の値と勾配を求める
for i in range(512):
    pil_conv_layer_output_value[:, :, i] *= pil_pooled_grads_value[i] # 勾配の平均をかけることで各クラスによる違いを明瞭にする
pil_heatmap=np.mean(pil_conv_layer_output_value, axis=-1)
# ヒートマップを作成
pil_heatmap = np.maximum(pil_heatmap, 0) # ReLUを通す
pil_heatmap /= np.max(pil_heatmap)
plt.matshow(pil_heatmap)
plt.show()

pil_img = cv2.imread("croppedmadrid4.jpg")
pil_heatmap = cv2.resize(pil_heatmap, (pil_img.shape[1], pil_img.shape[0])) #ヒートマップの幅と高さにリサイズ
pil_heatmap = np.uint8(255 * pil_heatmap)
pil_heatmap = cv2.applyColorMap(pil_heatmap, cv2.COLORMAP_JET) # 色を鮮やかにする
pil_superimposed_img = pil_heatmap * 0.4 + pil_img # ヒートマップと画像を足すことで重ねている
cv2.imwrite('pil_madrid_heatmap.jpg', pil_superimposed_img)
