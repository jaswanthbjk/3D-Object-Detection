from helpers.viz_util_for_lyft import PredViewer
from dataset.prepare_lyft_data_v2 import load_train_data
from test.test_data_loader import load_test_data
import matplotlib.pyplot as plt

# TODO: Need to add prepare_lyft_data_v2_rgb_test.FrustumRGBTestCase.test_plot_frustums to complete full pipeline

def plot_prediction_data():
    lyftd = load_train_data()

    pv = PredViewer(pred_file="./log_pred_v2_01.csv", lyftd=lyftd)

    # test_token = lyftd.sample[2]['token']
    test_token = pv.pred_pd.index[100]
    print(test_token)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    pv.render_camera_image(ax[0], sample_token=test_token, prob_threshold=0.1)

    pv.render_lidar_points(ax[1], sample_token=test_token, prob_threshold=0.1)

    fig.savefig("./camera_top_view.png", dpi=600)

    pv.render_3d_lidar_points_to_camera_coordinates(test_token, prob_threshold=0.1)


if __name__ == "__main__":
    plot_prediction_data()

    import mayavi.mlab as mlab

    mlab.savefig("./demo_3d_lidar_10.png")

    input()
