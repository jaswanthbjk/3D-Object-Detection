import pandas as pd

sample_csv = "/Users/kanhua/Downloads/3d-object-detection-for-autonomous-vehicles/sample_submission.csv"

pred_csv = "/Users/kanhua/Dropbox/Programming/lyft-3d-main/test_pred.csv"

sample_df = pd.read_csv(sample_csv, delimiter=',')

pred_df = pd.read_csv(pred_csv, delimiter=",")

print(pred_df.shape)
print("the beginning of the submissions template:")
print(sample_df.head())

print(sample_df.shape)

merged_df = pd.merge(left=sample_df, right=pred_df, how="left", on='Id')

print(merged_df.head())

print(merged_df.columns)

merged_df = merged_df.iloc[:, [0, 2]]

print(merged_df.head())

merged_df = merged_df.rename(columns={'PredictionString_y': 'PredictionString'})

merged_df.to_csv("prediction.csv", index=False)

print(merged_df.head())

print(merged_df.shape)
