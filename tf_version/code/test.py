# -*- encoding:utf-8 -*-
import os
import cv2
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from model import MultiModal


print("[INFO] 选择训练模型所在文件夹...")
dirId = input("dir id: ")
dirId = str(dirId)

# 加载训练好的模型
print("[INFO] 加载训练模型...")
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

print("[INFO] 加载测试数据...")
images = []
visits = []
def random_crop_and_normal(image_path, h=88, w=88):
    im = cv2.imread(image_path)
    height, width, _ = im.shape
    y = random.randint(1, height - h)
    x = random.randint(1, width - w)
    crop = im[y:y+h, x:x+w] / 255.0
    return crop

for i in range(10000):
    image_path = "../data/test_image/test/"+str(i).zfill(6)+".jpg"
    image = random_crop_and_normal(image_path)
    images.append(image)

    visit_path = "../data/npy/test_visit/"+str(i).zfill(6)+".npy"
    visit = np.load(visit_path)
    visit = visit / np.max(visit)
    visits.append(visit)

# 每次测试1000条数据，如果显存不够可以改小一些
print("[INFO] 开启测试...")
predictions = []
interval = 1000
for i in range(0, len(images), interval):
    predictions.extend(
        sess.run(
            tf.argmax(model.prediction, 1),
            feed_dict={
                model.image: images[i:i+interval],
                model.visit: visits[i:i+interval],
                model.training: False
            }
        )
    )
    print('[INFO] 第{}次测试结束!'.format(i+interval))

# 新建文件夹
if not os.path.exists("../result/"):
    os.mkdir("../result/")

# 将预测结果写入文件
with open("../result/result.txt", "w+") as f:
    for index, prediction in enumerate(predictions):
        f.write("%s \t %03d\n"%(str(index).zfill(6), prediction+1))
print("[INFO] 测试完成...")
