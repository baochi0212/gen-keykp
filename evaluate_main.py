from glob import glob
import os
import sys

output_merged_dir = sys.argv[1]

def eval(type, path):
    step_num = path.split("_")[-1]
    os.system(f"python infer_valid_set_llm.py {type} {path} ")
    os.system(f"bash ./evaluate_llm_valid.sh {type} {step_num}")
def test(type, path):
    step_num = path.split("_")[-1]
    os.system(f"python infer_llm.py {type} {path}")
    os.system(f"bash ./evaluate_llm.sh {type} {step_num}")
if __name__ == "__main__":
    for eval_path in glob(f"./lora_{output_merged_dir}*"):
        print("Evaluating checkpoint: ", eval_path)
        eval(output_merged_dir, eval_path)
        test(output_merged_dir, eval_path)
