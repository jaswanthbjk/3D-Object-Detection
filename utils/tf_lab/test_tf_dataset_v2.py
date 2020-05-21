import tensorflow as tf
from itertools import count


g=tf.Graph()
with g.as_default():
    dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
    iter = tf.compat.v1.data.make_one_shot_iterator(dataset)
    el = iter.get_next()
    el = iter.get_next()
    el2=tf.compat.v1.add(el,el)

    with tf.compat.v1.Session() as sess:
        try:
            for x in count():
                data_numpy=sess.run(el2)
                print(data_numpy)
        except tf.errors.OutOfRangeError:
                print("Process finished")
                pass