import tensorflow as tf
import os


def csvrd(filelist):
    # 1.构造文件队列
    file_queue = tf.train.string_input_producer(filelist)
    # 2.构造csv阅读器读取数据，默认行读取,用阅读器阅读，解包返回值接收,传入参数是文件队列
    print(file_queue)
    mycsvreader = tf.TextLineReader()
    label, value = mycsvreader.read(file_queue)
    print(label, value)
    # 3.对每一行内容解码，定义每一行的类型和缺失默认值
    records = [["None"], ["None"]]
    [exp, lab] = tf.decode_csv(value, record_defaults=records)
    # 4.想要读取多个数据就需要进行批处理
    value_batch, label_batch, exp_batch, lab_batch = tf.train.batch([value, label, exp, lab], batch_size=9, num_threads=1, capacity=9)
    return value_batch, label_batch, exp_batch, lab_batch


if __name__ == '__main__':
    # 找到文件，文件路径+文件名 放入列表
    path = "./data/csvdata/"
    file_name = os.listdir(path)  # listdir后面接路径，会把dir下的所有文件整理成一个list返回
    filelist = [os.path.join(path, file)for file in file_name]  # os.path.join后面传入两个参数，文件路径加文件名可以把他们串接成一个完整路径的文件名，
    # 接for循环成一个列表
    value_batch, label_batch, exp_batch, lab_batch = csvrd(filelist)
    with tf.Session() as sess:
        # 定义线程协调器
        coord = tf.train.Coordinator()
        # 开启读取文件的线程,返回的是线程
        threads = tf.train.start_queue_runners(sess, coord=coord)
        # 打印读取内容
        print(sess.run([value_batch, label_batch, exp_batch, lab_batch]))
        # 回收子线程
        coord.request_stop()
        coord.join(threads)
