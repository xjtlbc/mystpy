"""
实现单层全连接层对手写体的图形识别
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 定义命令行输入参数的方式
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean("is_train", False, "是否是训练")


def mnist():
    # 读取数据,注意这里默认的不是one_hot型的数据，如果是需要指定，one_hot=True
    mnist = input_data.read_data_sets("F:/python study base/sttensorflow/data/mnist/input_data/", one_hot=True)
    print(mnist)
    # 建立数据的占位符
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.int32, [None, 10])
    # 建立一个全连接网络， w[784, 10], b[10]
    with tf.variable_scope("model"):
        # 随机初始化权重和偏置
        weight = tf.Variable(tf.random_normal([784, 10], mean=0.0, stddev=1.0, name="w"))
        bias = tf.Variable(tf.random_normal([10], mean=0.0, stddev=1.0, name="b"))
        y_predict = tf.matmul(x, weight) + bias
    # 求出样本的损失值和平均值
    with tf.variable_scope("loss"):
        # 用求平均，交叉熵损失值的方式得到样本损失值的平均值
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_predict))

    # 梯度下降优化损失值更新权重,也就是训练过程
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    # 计算准确率
    with tf.variable_scope("acc"):
        # 根据预测值与目标值是否相等做成一个列表，相同为1，不同为0，求出列表的平均值即代表准确率.argmax,
        # 输入一个矩阵，制定维度，返回最大值所在的位置，用这种方式对两个y比较，相同返回1，不同返回0.
        equal_list = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_true, 1))  # [0, 1, 0, 0 , 1.....]
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))  # 在计算平均值之前先转换成float类型
    # 收集变量，单个变量用scalar
    tf.summary.scalar("losses", loss)
    tf.summary.scalar("acc", accuracy)
    # 收集高纬度变量
    tf.summary.histogram("w", weight)
    tf.summary.histogram("b", bias)
    # 定义合并op
    merged = tf.summary.merge_all()
    # 开启变量定义op
    va_init_op = tf.global_variables_initializer()
    # 定义保存模型op
    saver = tf.train.Saver()
    # 开启会话训练
    with tf.Session() as sess:
        # 初始化变量
        sess.run(va_init_op)
        # 建立events文件
        filewriter = tf.summary.FileWriter("./summary/test/", graph=sess.graph)
        acc_time = 0
        if FLAGS.is_train == True:
            for i in range(3000):
                if acc_time >= 3:
                    saver.save(sess, "./model/mnist")
                    break
                # 取出x，y
                x_mnist, y_mnist = mnist.train.next_batch(50)
                # 运行train_op
                sess.run(train_op, feed_dict={x: x_mnist, y_true: y_mnist})
                summary = sess.run(merged, feed_dict={x: x_mnist, y_true: y_mnist})
                filewriter.add_summary(summary, i)
                acc = sess.run(accuracy, feed_dict={x: x_mnist, y_true: y_mnist})
                print("第{}步的准确率为{}".format(i, acc))
                if i % 100 == 0:
                    saver.save(sess, "./model/mnist")
                if acc >= 0.9:
                    saver.save(sess, "./model/mnist")
                    acc_time += 1
        else:
            # 加载模型， 做出预测
            saver.restore(sess, "./model/mnist")
            predict_list = []
            for i in range(100):
                x_test, y_test = mnist.train.next_batch(1)
                y_test_true = tf.argmax(y_test, 1).eval()
                y_test_predict_one_hot = sess.run(y_predict, feed_dict={x:x_test, y_true: y_test})
                y_test_predict = tf.argmax(y_test_predict_one_hot, 1).eval()
                once_predict_res = tf.cast(tf.equal(y_test_true, y_test_predict).eval(), tf.float32)
                print("第{}个预测结果：{}，实际：{}，预测正确与否：{}".format(i, y_test_predict, y_test_true, once_predict_res.eval()))
                predict_list.append(once_predict_res)
            print(predict_list)
            print("准确率为{}".format(tf.reduce_mean(predict_list).eval()))

    return None


if __name__ == '__main__':
    mnist()
