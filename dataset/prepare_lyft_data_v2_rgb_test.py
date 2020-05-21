import unittest
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
from dataset.prepare_lyft_data_v2 import FrustumGenerator, get_all_boxes_in_single_scene, \
    parse_frustum_point_record, load_train_data, FrustumPoints2D, parse_frustum_point_record_2d
import matplotlib.pyplot as plt
import numpy as np
from test.test_data_loader import load_test_data
from helpers.object_classifier import TLClassifier, draw_result
from skimage.io import imread


class FrustumRGBTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.level5testdata = load_test_data()
        self.object_classifier = TLClassifier()

    def test_write_frustum_object(self):
        dummy_pc = np.zeros((5, 3), dtype=np.float32)
        fp2 = FrustumPoints2D(lyftd=self.level5testdata, point_cloud_in_box=dummy_pc, box_2d_pts=np.array([1, 2, 3, 4]),
                              frustum_angle=0.2, sample_token='sample_token',
                              camera_token='cam_token', score=1.0, object_name='car')

        example = fp2.to_train_example()
        example_proto_str = example.SerializeToString()
        example_tensors = parse_frustum_point_record_2d(example_proto_str)
        self.assertTrue(np.allclose(example_tensors['frustum_point_cloud'].numpy(), fp2.point_cloud_in_box))

    def test_one_sample_token(self):
        test_sample_token = self.level5testdata.sample[0]['token']

        print(test_sample_token)

        fg = FrustumGenerator(sample_token=test_sample_token, lyftd=self.level5testdata)

        fp = next(fg.generate_frustums_from_2d(self.object_classifier))

        example = fp.to_train_example()
        example_proto_str = example.SerializeToString()
        example_tensors = parse_frustum_point_record(example_proto_str)

        self.assertTrue(np.allclose(example_tensors['frustum_point_cloud'].numpy(), fp.point_cloud_in_box))

        # assert np.allclose(example_tensors['rot_box_3d'].numpy(), fp._get_rotated_box_3d())  # (8,3))

    def test_one_scene(self):
        print("writing one scene:")
        with tf.io.TFRecordWriter("scene1_test.tfrec") as tfrw:
            for fp in get_all_boxes_in_single_scene(0, from_rgb_detection=True,
                                                    ldf=self.level5testdata,
                                                    object_classifier=self.object_classifier):
                tfexample = fp.to_train_example()
                tfrw.write(tfexample.SerializeToString())

    def test_plot_frustums(self):
        from helpers.viz_util_for_lyft import PredViewer
        pv = PredViewer(pred_file="prediction.csv", lyftd=self.level5testdata)

        # test_token = lyftd.sample[2]['token']
        test_sample_token = pv.pred_pd.index[1]

        # test_sample_token = self.level5testdata.sample[2]['token']

        print(test_sample_token)

        fg = FrustumGenerator(sample_token=test_sample_token, lyftd=self.level5testdata, use_multisweep=True)

        ax_dict = {}
        for fp in fg.generate_frustums_from_2d(self.object_classifier):  # iterate through all the frustums
            if fp.camera_token not in ax_dict.keys():
                # Do the following plots when a camera image is first encountered
                fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21, 7))
                fig2, ax2d = plt.subplots()
                ax_dict[fp.camera_token] = (fig, ax)
                fp.render_image(ax[0])

                image_path = self.level5testdata.get_sample_data_path(fp.camera_token)

                image_array = imread(image_path)
                nboxes = fg.all_sel_boxes
                n_image_array = draw_result(image_array, nboxes)
                ax2d.imshow(n_image_array)

                channel = self.level5testdata.get("sample_data", fp.camera_token)['channel']
                fig2.savefig("./artifact/{}_2d_detection.png".format(channel), dpi=450)

            else:
                ax = ax_dict[fp.camera_token][1]

            fp.render_point_cloud_on_image(ax[0])

            fp.render_point_cloud_top_view(ax[1])

            fp.render_rotated_point_cloud_top_view(ax[2])

        for key in ax_dict.keys():
            fig, ax = ax_dict[key]
            channel = self.level5testdata.get("sample_data", key)['channel']
            fig.savefig("./artifact/{}_from_rgb.png".format(channel), dpi=450)


if __name__ == '__main__':
    unittest.main()
