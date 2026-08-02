import argparse
import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Parse command-line parameters early and set the CUDA device environment variable.
parser = argparse.ArgumentParser(description="Run the refactoring pipeline on a CSV file.")
parser.add_argument("--start-row", type=int, default=0, help="Row number (0-indexed) from which to start processing")
parser.add_argument("--cuda-device", type=str, required=True, help="CUDA device to use (e.g., 'cuda:5')")
parser.add_argument("--output-file", type=str, required=True, help="Path of output csv file.")
parser.add_argument(
    "--input-csv",
    type=str,
    default=str(PROJECT_ROOT / "data" / "dataset.csv"),
    help="Path to input CSV file."
)
parser.add_argument(
    "--initial-refactor-mode",
    type=str,
    choices=["generate", "precomputed"],
    default="generate",
    help="Use 'generate' to run initial StarCoder2 refactor, or 'precomputed' to use an existing iter1 file."
)
parser.add_argument(
    "--pre-refactored-dir",
    type=str,
    default=str(PROJECT_ROOT / "refactored_methods"),
    help="Directory containing precomputed *_iter1.java files when --initial-refactor-mode=precomputed."
)
args, _ = parser.parse_known_args()
os.environ["CUDA_DEVICE"] = args.cuda_device

from dotenv import load_dotenv
from refactoring_orchestrator import GPT4RefactoringOrchestrator

load_dotenv(PROJECT_ROOT / ".env")


def _get_repo_name(row: dict) -> str:
    if row.get("repo"):
        return row["repo"]
    if row.get("repository"):
        return row["repository"]
    raise KeyError("Input row missing both 'repo' and 'repository' columns.")


def main():
    GPT_API_KEY = os.getenv("OPENAI_API_KEY")
    if not GPT_API_KEY:
        print("[ERROR] Please set your OPENAI_API_KEY environment variable.")
        sys.exit(1)

    CSV_FILE_PATH = args.input_csv
    OUTPUT_CSV_PATH = args.output_file

    # Now include up to 10 iterative fixes.
    fieldnames = [
        "repo",
        "method",
        "test_method",
        "first_method_pass",
        "starcoder2_pass",
    ] + [f"iteration_{i}_pass" for i in range(1, 11)]

    with open(OUTPUT_CSV_PATH, mode="w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        if args.start_row == 0:
            writer.writeheader()

        with open(CSV_FILE_PATH, mode="r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for index, row in enumerate(reader):
                if index < args.start_row:
                    continue

                repo_name = _get_repo_name(row)
                repo_url  = f"https://github.com/{repo_name}.git"
                method_path      = row["method"]
                test_method_path = row["test_method"]
                method_name = os.path.splitext(os.path.basename(method_path))[0]

                if args.initial_refactor_mode == "precomputed":
                    sc2_iter1 = os.path.join(args.pre_refactored_dir, f"{method_name}_iter1.java")
                    if not (os.path.isfile(sc2_iter1) and os.path.getsize(sc2_iter1) > 0):
                        print(f"[WARNING] Missing or empty precomputed file {sc2_iter1}; skipping row {index}.")
                        continue

                print(f"\n\n=== Processing {method_name} in {method_path} ===")

                orchestrator = GPT4RefactoringOrchestrator(
                    gpt_api_key=GPT_API_KEY,
                    repo_name=repo_name,
                    repo_url=repo_url,
                    max_iterations=10
                )

                try:
                    results = orchestrator.run_refactoring_pipeline(
                        method_path=method_path,
                        method_name=method_name,
                        test_path=test_method_path,
                        initial_refactor_mode=args.initial_refactor_mode,
                        pre_refactored_dir=args.pre_refactored_dir,
                        language="java"
                    )
                except FileNotFoundError as e:
                    print(f"[WARNING] File not found: {e}. Skipping this row.")
                    continue

                # Initialize the output row with pass/fail fields set to N/A.
                row_dict = {
                    "repo": repo_name,
                    "method": method_path,
                    "test_method": test_method_path,
                    "first_method_pass": "N/A",
                    "starcoder2_pass":   "N/A",
                }
                # Initialize iteration_1_pass ... iteration_10_pass to N/A.
                for i in range(1, 11):
                    row_dict[f"iteration_{i}_pass"] = "N/A"

                # Fill the fields from the recorded run logs.
                for entry in results.get("logs", []):
                    i = entry.get("iteration", -1)
                    # find test_execution
                    pass_val = next((a.get("pass") for a in entry.get("actions", [])
                                     if a.get("step") == "test_execution"), None)
                    if pass_val == "cuda":
                        value_str = "cuda"
                    elif isinstance(pass_val, bool):
                        value_str = "Yes" if pass_val else "No"
                    else:
                        continue

                    if i == 0:
                        row_dict["first_method_pass"] = value_str
                    elif i == 1:
                        row_dict["starcoder2_pass"] = value_str
                    # iterative fixes 2→11 → iteration_1_pass…iteration_10_pass
                    elif 2 <= i <= 11:
                        idx = i - 1
                        row_dict[f"iteration_{idx}_pass"] = value_str

                writer.writerow(row_dict)
                output_file.flush()

                if results["success"]:
                    print(f"[SUCCESS] after {results['iterations']} iteration(s).")
                else:
                    print("[FAILURE] Could not fix within max_iterations.")

                for log in results.get("logs", []):
                    print(log)

    print(f"\nAll done! Results in: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
