from utils import ensemble_submissions

submission_dir = "."
# submission_dir = ["results/exploss", "results/mseloss", "results/linearloss", "submissions/0410"]
output_path = "ensemble_submission.csv"

ensemble_submissions(submission_dir, output_path)
