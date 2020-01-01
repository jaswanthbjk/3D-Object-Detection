from lyftdatasethelper import LyftDatasetHelper

level5data = LyftDatasetHelper(data_path=r'F:\\LyftDataset\\v1.01-train',
                         json_path=r'F:\\LyftDataset\\v1.01-train\\v1.01-train', verbose=True)
idx = 0
for scene in level5data.scene:
    cur_sample_token = scene["first_sample_token"]

    while cur_sample_token:
        print("[{0}] Current sample token: {1}".format(idx, cur_sample_token))
        sd_rec = level5data.get('sample_data', cur_sample_token)
        s_rec = level5data.get('sample', sd_rec['sample_token'])
        my_sample = level5data.get('sample', cur_sample_token)
        ann_rec = level5data.get('image_annotations', cur_sample_token)
        print(ann_rec)
        cur_sample_token = my_sample['next']
    idx += 1