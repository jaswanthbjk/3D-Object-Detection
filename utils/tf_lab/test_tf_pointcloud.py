import numpy as np
import tensorflow as tf


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

keys_to_feature={'point_cloud_3d':tf.FixedLenFeature([5],tf.float32),
                 'token':tf.VarLenFeature(tf.string)}

g=tf.Graph()
with g.as_default():

    test_pc=np.array([[8, 3, 0] ,[8, 2, 1]])
    test_token="AGSWDF234"
    feature_dict={'point_cloud_3d':float_list_feature(test_pc.ravel()),'token':bytes_feature(test_token.encode('utf8'))}
    tf_example=tf.train.Example(features=tf.train.Features(feature=feature_dict))
    example_message = tf_example.SerializeToString()
    parsed_example = tf.io.parse_single_example(example_message, keys_to_feature)
    #parsed_pc=tf.reshape(parsed_example['point_cloud_3d'],(2,3))
    parsed_pc=parsed_example['point_cloud_3d']
    #parsed_token=parsed_example['token']
    #parsed_token=parsed_example['token'].values[0].decode('UTF-8')
    #parsed_token2=tf.sparse_tensor_to_dense(parsed_example['token'], default_value=0)
    #parsed_token=tf.compat.v1.strings.unicode_decode(parsed_example['token'].values[0],input_encoding='bytes')
    with tf.compat.v1.Session() as sess:
        try:
            parsed_data=sess.run(parsed_pc)
            print(parsed_data)
        except tf.errors.OutOfRangeError:
            print("Process finished")
            pass