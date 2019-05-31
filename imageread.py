import tensorflow as tf
import os


def imageread(filelist):
    filequeue = tf.train.string_input_producer(filelist)
    myimagereader = tf.WholeFileReader()
    label, value = myimagereader.read(filequeue)
    print(value)
    image = tf.image.decode_jpeg(value)
    print(image)
    # image.resize_image这个函数只能修改长宽两个属性
    image_resize = tf.image.resize_images(image, [200, 200])
    # 重新制定通道数
    image_resize.set_shape([200, 200, 3])
    print(image_resize)
    # tf.train.batch里面第一个参数传入的一定要是一个列表
    image_batch = tf.train.batch([image_resize], batch_size=2, num_threads=1, capacity=5)
    print(image_batch)
    return image_batch


if __name__ == '__main__':
    path = "C:/Users/dingcacr/Pictures/2018-06/"
    filename = os.listdir(path)
    filelist = [os.path.join(path, file)for file in filename]
    image_batch = imageread(filelist)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)
        print(sess.run(image_batch))
        coord.request_stop()
        coord.join(threads)