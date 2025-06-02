import os
import glob
import pandas as pd
from tqdm import tqdm


def ensemble_submissions(submission_dir, output_path):
    if isinstance(submission_dir, str):
        submission_dir = [submission_dir]

    csv_files = []
    for dir in submission_dir:
        csv_files.extend(glob.glob(os.path.join(dir, "**", "*.csv"), recursive=True))

    if not csv_files:
        print("No CSV files found in the provided directory/directories.")
        return

    merged_df = None

    for _, file in tqdm(enumerate(csv_files), total=len(csv_files), desc="Ensembling submissions"):
        df = pd.read_csv(file)
        df = df.set_index("index")
        if merged_df is None:
            merged_df = df
        else:
            merged_df += df

    ensembled_df = merged_df / len(csv_files)
    ensembled_df = ensembled_df.reset_index()

    ensembled_df.to_csv(output_path, index=False)
    print(f"Ensembled submission saved to {output_path}")


submission_dir = ["./models/0511-6F", "./models/0512-10F", "./models/0512-31F", "./models/0517-55F"]
output_path = "ensemble_submission.csv"


ensemble_submissions(submission_dir, output_path)
