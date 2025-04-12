from utils import ensemble_submissions

submission_dir = "10folds/linearloss"
output_path = "ensemble_submission.csv"

ensemble_submissions(submission_dir, output_path)
