import numpy as np
import tensorflow as tf


# a = tf.constant(5.0)
# b = tf.constant(9.0)
# c = tf.add(a, b)
# plt = tf.placeholder(tf.float32, [None, 3])
# with tf.Session() as sess:
#     print(sess.run(c))
#     print(c.graph)
#     print(a.graph)
#     print(plt.shape)
#     print(a.shape)
# ----------------------------------------------
# zeros32 = tf.zeros([3, 2], dtype=tf.float32)
#
# with tf.Session() as sess:
#     sess.run(zeros32)
#     print(zeros32.eval())
#     print(zeros32)
# ------------------------------------
# a = tf.cast(np.arange(6).reshape((2, 3))+1, tf.float32)
# with tf.Session() as sess:
#     print(a.eval())
#     sess.run(a)
# ------------------------------------------------------
consta1 = tf.constant([[1, 2, 3], [4, 5, 6]])
var1 = tf.Variable(tf.random_normal([2, 3], mean=1.0, stddev=1.0))
var_init_op = tf.global_variables_initializer()
a = tf.constant(3.0)
b = tf.constant(5.0)
c = tf.add(a, b)
with tf.Session() as sess:
    sess.run(var_init_op)
    filewrite = tf.summary.FileWriter('./summary/test/', graph=sess.graph)
    print(consta1.eval())
    print(var1.eval())
    print(sess.run(c))
