import cv2
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

label = cv2.imread("Algorithm/4_OUT.bmp", cv2.IMREAD_GRAYSCALE)
detection = cv2.imread("Algorithm/changemap.jpg", cv2.IMREAD_GRAYSCALE)
# 匹配shape保持一致
detection = cv2.resize(detection, (label.shape[1], label.shape[0]))
# 图像二值化
label_bin = np.where(label > 0, 1, 0)
detection_bin = np.where(detection > 0, 1, 0)
# 计算交集、并集
intersection = np.logical_and(label_bin, detection_bin)
union = np.logical_or(label_bin, detection_bin)

# 计算准确率、召回率、F1-score
tn, fp, fn, tp = confusion_matrix(label_bin.flatten(), detection_bin.flatten()).ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * precision * recall / (precision + recall)

# 输出结果 

print(accuracy)
print(precision)
print(recall)
print(f1_score)

# 计算
img1 = label_bin
img2 = detection_bin

# 计算混淆矩阵
confusion = confusion_matrix(img1.ravel(), img2.ravel())

# 计算总体精度
overall_accuracy = np.trace(confusion) / np.sum(confusion)

# 计算kappa系数
kappa = cohen_kappa_score(img1.ravel(), img2.ravel())

# 计算过检率和漏检率
false_alarm = confusion[1, 0] / np.sum(confusion[1, :])
missed_detection = confusion[0, 1] / np.sum(confusion[0, :])

# 输出结果
print('Overall accuracy: ', overall_accuracy)
print('Kappa coefficient: ', kappa)
print('False alarm rate: ', false_alarm)
print('Missed detection rate: ', missed_detection)
