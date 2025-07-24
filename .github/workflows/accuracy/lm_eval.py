from pathlib import Path
import yaml
from multiprocessing import Queue
import multiprocessing
import lm_eval
import numpy as np
import argparse
import gc
import json
import sys
import time

import lm_eval
import torch

RTOL = 0.03
ACCURACY_FLAG = {}

FILTER = {
    "gsm8k": "exact_match,flexible-extract",
    "ceval-valid": "acc,none",
    "mmmu_val": "acc,none",
}

def run_accuracy_test(queue, model, dataset, eval_config):
    try: 
        eval_params = {
            "model": eval_config.get("model_type"),
            "model_args": eval_config.get("model_args"),
            "tasks": dataset.get("name"),
            "apply_chat_template": eval_config.get("apply_chat_template"),
            "fewshot_as_multiturn": eval_config.get("fewshot_as_multiturn"),
            "batch_size": eval_config.get("batch_size")
        }
        if eval_config.get("model_type") == "vllm":
            eval_params["num_fewshot"] = 5 
        results = lm_eval.simple_evaluate(**eval_params)
        print(f"Success: {model} on {dataset} ")
        measured_value = results["results"]
        queue.put(measured_value)
    except Exception as e:
        print(f"Error in run_accuracy_test: {e}")
        queue.put(e)
        sys.exit(1)
    finally:
        if "results" in locals():
            del results
        gc.collect()
        torch.npu.empty_cache()
        time.sleep(5)
        
def generate_md(model_name, tasks_list, args, datasets):
    pass

def safe_md(args, accuracy, datasets):
    """
    Safely generate and save Markdown report from accuracy results.
    """
    data = json.loads(json.dumps(accuracy))
    for model_key, tasks_list in data.items():
        md_content = generate_md(model_key, tasks_list, args, datasets)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"create Markdown file:{args.output}")

def main(model):
    accuracy = {}
    accuracy[args.model] = []
    result_queue: Queue[float] = multiprocessing.Queue()
    current_dir = Path(__file__).parent
    config_path = current_dir / "workflows" / "accuracy" / f"{model}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        eval_config = yaml.safe_load(f)
    datasets = eval_config.get("tasks")
    datasets_str = ", ".join([task["name"] for task in eval_config])
    for dataset in datasets:
        ground_truth = dataset.get("ground_truth")
        p = multiprocessing.Process(
            target=run_accuracy_test, args=(result_queue, args.model, dataset, eval_config)
        )
        p.start()
        p.join()
        if p.is_alive():
            p.terminate()
            p.join()
        gc.collect()
        torch.npu.empty_cache()
        time.sleep(10)
        result = result_queue.get()
        print(result)
        if np.isclose(ground_truth, result[dataset.get("name")][FILTER[dataset.get("name")]], rtol=RTOL):
            ACCURACY_FLAG[dataset] = "✅"
        else:
            ACCURACY_FLAG[dataset] = "❌"
        accuracy[args.model].append(result)
        print(accuracy)
        safe_md(args, accuracy, datasets_str)
        
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(
        description="Run model accuracy evaluation and generate report"
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--vllm_ascend_version", type=str, required=False)
    parser.add_argument("--torch_version", type=str, required=False)
    parser.add_argument("--torch_npu_version", type=str, required=False)
    parser.add_argument("--vllm_version", type=str, required=False)
    parser.add_argument("--cann_version", type=str, required=False)
    parser.add_argument("--vllm_commit", type=str, required=False)
    parser.add_argument("--vllm_ascend_commit", type=str, required=False)
    args = parser.parse_args()
    main(args)                                               
    
    