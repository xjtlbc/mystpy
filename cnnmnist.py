import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean("is_train", False, "是否训练")


def w_init(shape):
    """
    点定义一个权重初始化函数，传入shape，用正态分布初始化
    :param shape:
    :return:
    """
    w = tf.Variable(tf.random_normal(shape, mean=0, stddev=1.0))
    return w


def bias_init(shape):
    """
    定义一个权重初始化函数，给定一个shape，用正态分布初始化
    :param shape:
    :return:
    """
    bias = tf.Variable(tf.random_normal(shape, mean=0.0, stddev=1.0))
    return bias


def model():
    """
    自定义的卷积模型，两层卷积一层全连接
    :return:
    """
    # 定义占位符
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.int32, [None, 10])
    # 第一层卷积
    with tf.variable_scope("con1"):
        # 卷积，激活，池化
        # 随机初始化权重w[5, 5, 1, 32], b[32]
        w_con1 = w_init([5, 5, 1, 32])
        b_con1 = bias_init([32])
        # 定义卷积,但是卷积之前要注意形状转换，以后建议在注释中标出大小
        # 转换形状[None, 784] ---> [None, 28, 28, 1]
        x_reshape_from_x = tf.reshape(x, [-1, 28, 28, 1])
        # 进行第一次卷积[None, 28, 28, 1] ---> [None, 28, 28, 32]
        x_con1 = tf.nn.conv2d(x_reshape_from_x, w_con1, strides=[1, 1, 1, 1], padding="SAME") + b_con1
        # 进行激活用relu， 提升网络的非线性能力
        x_relu1 = tf.nn.relu(x_con1)
        # 进行池化，提取特征，将数据量减小[None, 28, 28, 32] ---> [None, 14, 14, 32]
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # 第二层卷积
    with tf.variable_scope("con2"):
        # [None, 14, 14, 32] ---> [None, 14, 14, 64] ---> [None, 7, 7, 64]
        w_con2 = w_init([5, 5, 32, 64])
        b_con2 = bias_init([64])
        x_con2 = tf.nn.conv2d(x_pool1, w_con2, strides=[1, 1, 1, 1], padding="SAME")
        x_relu2 = tf.nn.relu(x_con2)
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    with tf.variable_scope("fc"):
        # 需要将形状变成[None, 7, 7, 64] ---> [None, 7*7*64] ---> [None, 10]
        w_fc = w_init([7*7*64, 10])
        b_fc = bias_init([10])
        x_fc_reshape_from_x_pool2 = tf.reshape(x_pool2, [-1, 7*7*64])
        y_predict = tf.matmul(x_fc_reshape_from_x_pool2, w_fc) + b_fc

    return x, y_true, y_predict


def cnnmnist():
    mnist = input_data.read_data_sets("F:/python study base/sttensorflow/data/mnist/input_data/", one_hot=True)
    print(mnist)
    x, y_true, y_predict = model()
    # 用模型返回的x, y_true, y_predict做交叉熵损失计算
    with tf.variable_scope("soft_cross"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_predict))
    # 梯度下降优化，训练过程
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
    with tf.variable_scope("acc"):
        equal_list = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))
    # 开启变量定义op
    var_init_op = tf.global_variables_initializer()
    # 保存模型的op
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(var_init_op)
        acc_time = 0
        if FLAGS.is_train:
            for i in range(10000):
                if acc_time >= 3:
                    saver.save(sess, "./model/cnnmnist/cnnmnist")
                    break
                x_mnist, y_mnist = mnist.train.next_batch(50)
                sess.run(train_op, feed_dict={x: x_mnist, y_true: y_mnist})
                acc = sess.run(accuracy, feed_dict={x: x_mnist, y_true: y_mnist})
                print("第{}步训练的准确率为：{}".format(i, acc))
                if i % 100 == 0:
                    saver.save(sess, "./model/cnnmnist/cnnmnist")
                if acc >= 0.9:
                    acc_time += 1
                    saver.save(sess, "./model/cnnmnist/cnnmnist")
        else:
            saver.restore(sess, "./model/cnnmnist/cnnmnist")
            predict_list = []
            for i in range(100):
                x_test, y_test = mnist.train.next_batch(1)
                y_test_true = tf.argmax(y_test, 1).eval()
                y_test_predict_one_hot = sess.run(y_predict, feed_dict={x: x_test, y_true: y_test})
                y_test_predict = tf.argmax(y_test_predict_one_hot, 1).eval()
                once_predict_res = tf.cast(tf.equal(y_test_true, y_test_predict).eval(), tf.float32)
                predict_list.append(once_predict_res)
                print("第{}次测试的实际值为：{}, 预测值为：{}, 预测正确与否：{}".format(i, y_test_true, y_test_predict, once_predict_res.eval()))
            print(predict_list)
            acc_test = tf.reduce_mean(predict_list).eval()
            print(acc_test)

    return None


if __name__ == '__main__':
    cnnmnist()