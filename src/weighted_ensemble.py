import os
import glob
import pandas as pd
from tqdm import tqdm


def ensemble_submissions(submission_dirs, output_path, weights=None):
    if isinstance(submission_dirs, str):
        submission_dirs = [submission_dirs]

    if weights is None:
        weights = [1.0] * len(submission_dirs)
    else:
        assert len(weights) == len(submission_dirs), "权重数量需与文件夹数量一致"

    folder_ensembles = []

    for dir in tqdm(submission_dirs, desc="Averaging each folder"):
        csv_files = glob.glob(os.path.join(dir, "**", "*.csv"), recursive=True)
        if not csv_files:
            print(f"No CSV files found in {dir}, skipping.")
            continue

        folder_df = None
        for file in csv_files:
            df = pd.read_csv(file)
            df = df.set_index("index")
            if folder_df is None:
                folder_df = df
            else:
                folder_df += df
        folder_df /= len(csv_files)
        folder_ensembles.append(folder_df)

    merged_df = None
    total_weight = 0
    for i, folder_df in enumerate(folder_ensembles):
        weight = weights[i]
        if merged_df is None:
            merged_df = folder_df * weight
        else:
            merged_df += folder_df * weight
        total_weight += weight

    ensembled_df = merged_df / total_weight
    ensembled_df = ensembled_df.reset_index()

    ensembled_df.to_csv(output_path, index=False)
    print(f"Ensembled submission saved to {output_path}")


submission_dir = ["./models/0511-6F", "./models/0512-10F", "./models/0512-31F"]
weights = [0.33, 0.34, 0.33]
output_path = "ensemble_submission.csv"

ensemble_submissions(submission_dir, output_path, weights)
