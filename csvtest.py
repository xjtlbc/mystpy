import tensorflow as tf
import os


def csvbatchs(filelist):
    filequeue = tf.train.string_input_producer(filelist)
    csvreader = tf.TextLineReader()
    label, value = csvreader.read(filequeue)
    record = [["None"], ["None"]]
    value1, value2 = tf.decode_csv(value, record_defaults=record)
    value1_batch, value2_batch = tf.train.batch([value1, value2], batch_size=9, num_threads=1, capacity=9)

    return  value1_batch, value2_batch


if __name__ == '__main__':
    path = "./data/csvdata/"
    filenames = os.listdir(path)
    filelist = [os.path.join(path, file)for file in filenames]
    value1_batch, value2_batch = csvbatchs(filelist)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)
        print(sess.run([value1_batch, value2_batch]))
        coord.request_stop()
        coord.join(threads)
