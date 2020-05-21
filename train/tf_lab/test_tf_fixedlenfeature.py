import numpy as np
import tensorflow as tf


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


keys_to_feature={'array':tf.FixedLenSequenceFeature([3],tf.float32,allow_missing=True)}

g=tf.Graph()
with g.as_default():

    test_pc=np.array([[8, 3, 0] ,[8, 2, 1]])
    feature_dict={'array':float_list_feature(test_pc.ravel())}
    tf_example=tf.train.Example(features=tf.train.Features(feature=feature_dict))


    # serialize and parse
    example_message = tf_example.SerializeToString()
    parsed_example = tf.io.parse_single_example(example_message, keys_to_feature)
    parsed_pc=parsed_example['array']

    with tf.compat.v1.Session() as sess:
        parsed_data=sess.run(parsed_pc)
        print(parsed_data)
