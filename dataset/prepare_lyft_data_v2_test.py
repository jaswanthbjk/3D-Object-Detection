import tensorflow as tf

tf.compat.v1.enable_eager_execution()
from dataset.prepare_lyft_data_v2 import FrustumGenerator, get_all_boxes_in_single_scene, \
    parse_frustum_point_record, load_train_data
import matplotlib.pyplot as plt
import numpy as np

level5data = load_train_data()


def test_one_sample_token():
    test_sample_token = level5data.sample[0]['token']

    print(test_sample_token)

    fg = FrustumGenerator(sample_token=test_sample_token, lyftd=level5data)

    fp = next(fg.generate_frustums())

    example = fp.to_train_example()
    example_proto_str = example.SerializeToString()

    example_tensors = parse_frustum_point_record(example_proto_str)

    assert np.allclose(example_tensors['frustum_point_cloud'].numpy(), fp.point_cloud_in_box)

    assert np.allclose(example_tensors['rot_box_3d'].numpy(), fp._get_rotated_box_3d())  # (8,3))


def test_write_tfrecord():
    test_sample_token = level5data.sample[0]['token']

    print(test_sample_token)

    fg = FrustumGenerator(sample_token=test_sample_token, lyftd=level5data)

    with tf.io.TFRecordWriter("./unit_test_data/test.tfrec") as tfrw:
        for fp in fg.generate_frustums():
            tfexample = fp.to_train_example()
            tfrw.write(tfexample.SerializeToString())


def test_plot_one_frustum():
    test_sample_token = level5data.sample[0]['token']

    print(test_sample_token)

    fg = FrustumGenerator(sample_token=test_sample_token, lyftd=level5data)

    ax_dict = {}
    for fp in fg.generate_frustums():
        if fp.camera_token not in ax_dict.keys():
            fig, ax = plt.subplots(1, 3)
            ax_dict[fp.camera_token] = (fig, ax)
            fp.render_image(ax[0])
        else:
            ax = ax_dict[fp.camera_token][1]

        fp.render_point_cloud_on_image(ax[0])

        fp.render_point_cloud_top_view(ax[1])

        fp.render_rotated_point_cloud_top_view(ax[2])

    for key in ax_dict.keys():
        fig, ax = ax_dict[key]
        channel = level5data.get("sample_data", key)['channel']
        fig.savefig("./artifact/{}.png".format(channel))


def test_one_scene():
    print("writing one scene:")
    with tf.io.TFRecordWriter("./unit_test_data/scene1_test.tfrec") as tfrw:
        for fp in get_all_boxes_in_single_scene(0, False, level5data):
            tfexample = fp.to_train_example()
            tfrw.write(tfexample.SerializeToString())


def test_load_example():
    test_write_tfrecord()

    filenames = ['./unit_test_data/test.tfrec']
    raw_dataset = tf.data.TFRecordDataset(filenames)
    for raw_record in raw_dataset.take(5):
        example = parse_frustum_point_record(raw_record)
        print("box_2d:", example['box_2d'].numpy())
        print("type_name:", example["type_name"].numpy().decode('utf8'))
        print("seg_label", example["seg_label"].numpy())


def test_tfdataset():
    def parse_data(raw_record):
        example = parse_frustum_point_record(raw_record)
        return example['seg_label']
        # return #example['rot_frustum_point_cloud'], tf.cast(example['one_hot_vec'], tf.float32), \
        # tf.cast(example['seg_label'], tf.int32), \
        # example['rot_box_center'], \
        # tf.cast(example['rot_angle_class'], tf.int32), \
        # example['rot_angle_residual'], \
        # tf.cast(example['size_class'], tf.int32), \
        # example['size_residual']

    filenames = ['./unit_test_data/test.tfrec']
    full_dataset = tf.data.TFRecordDataset(filenames)
    parsed_dataset = full_dataset.map(parse_data)

    for batched_data in parsed_dataset.batch(3):
        print(batched_data)


from dataset.prepare_lyft_data_v2 import parse_inference_data


def test_parse_inference_data():
    filenames = ['./unit_test_data/test.tfrec']
    full_dataset = tf.data.TFRecordDataset(filenames)
    parsed_dataset = full_dataset.map(parse_inference_data)

    for batched_data in parsed_dataset.take(1):
        print(batched_data)


# test_one_sample_token()
test_plot_one_frustum()
#test_write_tfrecord()
# test_one_scene()

#test_load_example()

# test_tfdataset()

# test_parse_inference_data()
