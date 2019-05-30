# -*- encoding:utf-8 -*-
from model import MultiModal
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2

# 选择gpu设备
# deviceId = input("device id: ")
# os.environ["CUDA_VISIBLE_DEVICES"] = deviceId
# 选择文件夹
dirId = input("dir id: ")
dirId = str(dirId)

# 加载训练好的模型
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
with sess.graph.as_default():
    with sess.as_default():
        model = MultiModal()
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        var_list = [var for var in tf.global_variables() if "moving" in var.name]
        var_list += [var for var in tf.global_variables() if "global_step" in var.name]
        var_list += tf.trainable_variables()
        saver = tf.train.Saver(var_list=var_list, max_to_keep=1)
        last_file = tf.train.latest_checkpoint("../model/"+dirId)
        if last_file:
            tf.logging.info('Restoring model from {}'.format(last_file))
            saver.restore(sess, last_file)


# 载入所有valid数据
images = []
visits = []
labels = []
NPYROOT = '../data/npy/train_visit/'
with open('../data/valid.txt') as fi:
    for line in fi.readlines():
        img_name = line.split('/')[-1]
        image = cv2.imread(line)[0:88,0:88,:]
        images.append(image)
        visit = np.load(os.path.sep.join(NPYROOT, img_name.replace('jpg', 'npy')))
        visits.append(visit)
        labels.append(int(line.split('/')[-2]))

predictions = []
# 每次测试1000条数据，如果显存不够可以改小一些
SUMV = len(labels)
STEP = 50
for i in range(0, SUMV, STEP):
    predictions.extend(sess.run(tf.argmax(model.prediction, 1),
                          feed_dict={model.image: images[i:i+STEP],
                                     model.visit: visits[i:i+STEP],
                                     model.training: False}))
    print('第%d次完成' % i)

# 新建文件夹
if not os.path.exists("../result/"):
    os.mkdir("../result/")

# 将预测结果写入文件
num = 0
with open("../result/valid.txt", "w+") as f:
    for index, prediction in enumerate(predictions):
        prediction = prediction + 1
        f.write("%s \t %03d\n"%(str(index).zfill(6), prediction))
        if prediction == labels[index]:
            num += 1

print("验证集图像数量: %d" % SUMV)
print("验证通过数量  : %d" % num)
print("验证集准确率  : %f" % (1.0*num/SUMV))
print("验证完成!")
