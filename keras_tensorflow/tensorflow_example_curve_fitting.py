# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:48:15 2017

@author: Xianhui
"""

import tensorflow as tf
import numpy as np

#%% set up calculation diagram
x_data = np.random.rand(100).astype('float32')
y_data = x_data * 0.1 + 0.3

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(loss=loss)

init = tf.global_variables_initializer()

#%% now it run the session
sess = tf.Session()
sess.run(init)
print('initial W b ', sess.run(W), sess.run(b))

for step in range(200):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
