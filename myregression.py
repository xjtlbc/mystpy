import os
import tensorflow as tf

with tf.variable_scope("data"):
    # 1.准备数据
    x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name="x_data")
    y_true = tf.matmul(x, [[0.7]],) + 0.8

with tf.variable_scope("model"):
    # 2.建立回归模型
    weight = tf.Variable(tf.random_normal([1, 1], mean=1.0, stddev=0.5, name="w"))
    bias = tf.Variable(tf.random_normal([1, 1], mean=1.0, stddev=0.5, name="b"))
    y_predict = tf.matmul(x, weight) + bias

with tf.variable_scope("loss"):
    # 3.建立损失函数，计算均方误差
    loss = tf.reduce_mean(tf.square((y_true - y_predict)))

with tf.variable_scope("optimize_op"):
    # 4.梯度优化损失
    optimi_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 初始化变量op
var_init_op = tf.global_variables_initializer()
# 收集需要显示的张量
tf.summary.scalar("losses", loss)
tf.summary.histogram("w", weight)
tf.summary.histogram("biases", bias)
# 定义合并变量的op
merged = tf.summary.merge_all()
saver = tf.train.Saver()
tf.app.flags.DEFINE_integer("max_step", 500, "模型最大训练次数")
tf.app.flags.DEFINE_string("model_dir", "./model/model", "模型保存路径")
FLAGS = tf.app.flags.FLAGS


# 开启会话
with tf.Session() as sess:
    sess.run(var_init_op)
    # 打印随机初始化的值
    print("初始化的权重和偏置为:{}{}".format(weight.eval(), bias.eval()))
    file_write = tf.summary.FileWriter('./summary/test/', graph=sess.graph)
    if os.path.exists(FLAGS.model_dir):
        saver.restore(sess, FLAGS.model_dir)
    for i in range(FLAGS.max_step):
        sess.run(optimi_op)
        # 打印优化后的权重和偏置
        # 每次迭代都要运行合并的op
        summary = sess.run(merged)
        file_write.add_summary(summary)
        print("第{}优化后的权重和偏置为: {}{}".format(i, weight.eval(), bias.eval()))
    saver.save(sess, FLAGS.model_dir)