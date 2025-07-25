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

def load_model_config(model):
    """Load model configuration from YAML file."""
    config_path = "./benchmarks/accuracy" / f"{model}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        eval_config = yaml.safe_load(f)
    return eval_config

def run_accuracy_test(queue, model, dataset):
    try: 
        eval_config = load_model_config(model)
        eval_params = {
            "model": eval_config.get("model_type"),
            "model_args": eval_config.get("model_args"),
            "tasks": dataset.get("name"),
            "apply_chat_template": eval_config.get("apply_chat_template"),
            "fewshot_as_multiturn": eval_config.get("fewshot_as_multiturn"),
            "batch_size": eval_config.get("dataset").get("batch_size"),
            "num_fewshot": eval_config.get("num_fewshot"),
        }
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
    """Generate Markdown report with evaluation results"""
    # Format the run command
    model = model_name.split("/")[1]
    eval_config = load_model_config(model)
    model_type = eval_config.get("model_type")
    pretrained = eval_config.get("model_args").get("pretrained")
    max_model_len = eval_config.get("model_args").get("max_model_len")
    dtype = eval_config.get("model_args").get("dtype")
    dtype = eval_config.get("model_args").get("tensor_parallel_size")
    gpu_memory_utilization = eval_config.get("model_args").get("gpu_memory_utilization")
    num_fewshot = eval_config.get("num_fewshot")
    
    if eval_config.get("apply_chat_template") == True:
        apply_chat_template = "--apply_chat_template"
    else:
        apply_chat_template = ""
        
    if eval_config.get("fewshot_as_multiturn") == True:
        fewshot_as_multiturn = "--fewshot_as_multiturn"
    else:
        fewshot_as_multiturn = ""
    
    run_cmd = (
        f"export MODEL_ARGS='pretrained={pretrained},max_model_len={max_model_len},dtype={dtype},"
        f"tensor_parallel_size={tensor_parallel_size},gpu_memory_utilization={gpu_memory_utilization}'\n"
        f"lm_eval --model {model_type} --model_args $MODEL_ARGS --tasks {datasets} \\\n"
        f"{apply_chat_template} {fewshot_as_multiturn} {num_fewshot} --batch_size 1"
    )

    # Version information section
    version_info = (
        f"**vLLM Version**: vLLM: {args.vllm_version} "
        f"([{args.vllm_commit}]({VLLM_URL + args.vllm_commit})), "
        f"vLLM Ascend: {args.vllm_ascend_version} "
        f"([{args.vllm_ascend_commit}]({VLLM_ASCEND_URL + args.vllm_ascend_commit}))  "
    )
    
    # Report header with system info
    preamble = f"""
        # {model}

        {version_info}

        **Software Environment**:  
        - CANN: {args.cann_version}  
        - PyTorch: {args.torch_version}  
        - torch-npu: {args.torch_npu_version}  

        **Hardware Environment**:  
        - Atlas A2 Series  

        **Datasets**:  
        {datasets}  

        **Parallel Mode**:  
        {eval_config.get("parallel_mode")}  

        **Execution Mode**:  
        {eval_config.get("execution_mode")}  

        **Command**:  
        ```bash
        {run_cmd}
        ```
    """

    header = (
        "| Task                  | Filter | n-shot | Metric   | Value   | Stderr |\n"
        "|-----------------------|-------:|-------:|----------|--------:|-------:|"
    )
    rows = []
    rows_sub = []
    # Process results for each task
    for task_dict in tasks_list:
        for key, stats in task_dict.items():
            alias = stats.get("alias", key)
            task_name = alias.strip()
            if "exact_match,flexible-extract" in stats:
                metric_key = "exact_match,flexible-extract"
            else:
                metric_key = None
                for k in stats:
                    if "," in k and not k.startswith("acc_stderr"):
                        metric_key = k
                        break
            if metric_key is None:
                continue
            metric, flt = metric_key.split(",", 1)
    
            value = stats[metric_key]
            stderr = stats.get(f"{metric}_stderr,{flt}", 0)
            flag = ACCURACY_FLAG.get(task_name, "")
            row = (
                f"| {task_name:<37} "
                f"| {flt:<6} "
                f"| {num_fewshot:6} "
                f"| {metric:<6} "
                f"| {flag}{value:>5.4f} "
                f"| ± {stderr:>5.4f} |"
            )
            if not task_name.startswith("-"):
                rows.append(row)
                rows_sub.append(
                    "<details>"
                    + "\n"
                    + "<summary>"
                    + task_name
                    + " details"
                    + "</summary>"
                    + "\n" * 2
                    + header
                )
            rows_sub.append(row)
        rows_sub.append("</details>")
    # Combine all Markdown sections
    md = (
        preamble
        + "\n"
        + header
        + "\n"
        + "\n".join(rows)
        + "\n"
        + "\n".join(rows_sub)
        + "\n"
    )
    print(md)
    return md

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


def main(args):
    accuracy = {}
    accuracy[args.model] = []
    result_queue: Queue[float] = multiprocessing.Queue()
    eval_config = load_model_config(args.model)
    datasets = eval_config.get("tasks")
    datasets_str = ", ".join([task["name"] for task in eval_config])
    for dataset in datasets:
        dataset_name = dataset.get("name")
        ground_truth = dataset.get("ground_truth")
        p = multiprocessing.Process(
            target=run_accuracy_test, args=(result_queue, args.model, dataset)
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
        if np.isclose(ground_truth, result[dataset_name][FILTER[dataset_name]], rtol=RTOL):
            ACCURACY_FLAG[dataset_name] = "✅"
        else:
            ACCURACY_FLAG[dataset_name] = "❌"
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
    