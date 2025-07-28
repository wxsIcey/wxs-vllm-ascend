from dataclasses import dataclass
from pathlib import Path
import yaml
import numpy as np
from jinja2 import Environment, FileSystemLoader
import lm_eval

RTOL = 0.03


@dataclass
class EnvConfig:
    vllm_version: str
    vllm_commit: str
    vllm_ascend_version: str
    vllm_ascend_commit: str
    cann_version: str
    torch_version: str
    torch_npu_version: str
    
    
@pytest.fixture
def env_config():
    return EnvConfig(
        vllm_version=os.getenv('VLLM_VERSION'),
        vllm_commit=os.getenv('VLLM_COMMIT'),
        vllm_ascend_version=os.getenv('VLLM_ASCEND_VERSION'),
        vllm_ascend_commit=os.getenv('VLLM_ASCEND_COMMIT'),
        cann_version=os.getenv('CANN_VERSION'),
        torch_version=os.getenv('TORCH_VERSION'),
        torch_npu_version=os.getenv('TORCH_NPU_VERSION')
    )
    
    
# Qwen/Qwen3-30B-A3B有一个模型参数为enable_expert_parallel=True
# Qwen/Qwen2.5-VL-7B-Instruct"有一个模型参数为max_images=2
# Qwen2.5-VL-7B-Instruct的执行命令不包含--num_fewshot
# 这些参数像是为了达到好的结果做的调整，暂不考虑?
def build_model_args(eval_config, tp_size):
    trust_remote_code = eval_config.get("trust_remote_code", False)
    max_model_len = eval_config.get("max_model_len", 4096)
    return (
        f"pretrained={eval_config['model_name']},"
        f"tensor_parallel_size={tp_size},"
        f"enforce_eager=true,"
        f"add_bos_token=true,"
        f"trust_remote_code={trust_remote_code},"
        f"max_model_len={max_model_len}"
    )


def launch_lm_eval(eval_config, tp_size):
    model_args = build_model_args(eval_config, tp_size)
    results = lm_eval.simple_evaluate(
        model=eval_config.get("model", "vllm"),
        model_args=model_args,
        tasks=[task["name"] for task in eval_config["tasks"]],
        num_fewshot=eval_config["num_fewshot"],
        limit=eval_config["limit"],
        batch_size="auto",
    )
    return model_args, results


def generate_report(tp_size, eval_config, report_data, report_template, output_path, env_config: EnvConfig):
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(report_template)
    model_args = build_model_args(eval_config, tp_size)
    datasets = ", ".join([task["name"] for task in eval_config["tasks"]])
    report_content = template.render(
        vllm_version=env_config.vllm_version,
        vllm_commit=env_config.vllm_commit,
        vllm_ascend_version=env_config.vllm_ascend_version,
        vllm_ascend_commit=env_config.vllm_ascend_commit,
        cann_version=env_config.cann_version,
        torch_version=env_config.torch_version,
        torch_npu_version=env_config.torch_npu_version,
        model_name=eval_config["model_name"],
        model_args=model_args,
        model_type=eval_config["model"],
        datasets=datasets,
        num_fewshot=eval_config["num_fewshot"],
        limit=eval_config["limit"],
        batch_size=eval_config["batch_size"],
        rows=report_data.rows
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        

def test_lm_eval_correctness_param(config_filename, tp_size, report_template, output_path, env_config: EnvConfig):
    eval_config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))
    results = launch_lm_eval(eval_config, tp_size)
    success = True
    report_data = {"rows": []}
    
    for task in eval_config["tasks"]:
        for metric in task["metrics"]:
            ground_truth = metric["value"]
            measured_value = results["results"][task["name"]][metric["name"]]
            print(f"{task['name']} | {metric['name']}: "
                  f"ground_truth={ground_truth} | measured={measured_value}")
            success = success and np.isclose(
                ground_truth, measured_value, rtol=RTOL)

            report_data.rows.append({
                "task": task["name"],
                "metric": metric["name"],
                "n_shot": eval_config["num_fewshot"],
                "value": f"✅{measured_value}" if is_success else f"❌{measured_value}",
                "stderr": results["results"][task["name"]].get("stderr", 0.0)
            })
    generate_report(tp_size, eval_config, report_data, report_template, output_path, env_config: EnvConfig)
    assert success
    
