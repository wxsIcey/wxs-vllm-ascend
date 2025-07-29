import os
from dataclasses import dataclass

import lm_eval
import numpy as np
import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

RTOL = 0.02


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
def env_config() -> EnvConfig:
    return EnvConfig(vllm_version=os.getenv('VLLM_VERSION', ''),
                     vllm_commit=os.getenv('VLLM_COMMIT', ''),
                     vllm_ascend_version=os.getenv('VLLM_ASCEND_VERSION', ''),
                     vllm_ascend_commit=os.getenv('VLLM_ASCEND_COMMIT', ''),
                     cann_version=os.getenv('CANN_VERSION', ''),
                     torch_version=os.getenv('TORCH_VERSION', ''),
                     torch_npu_version=os.getenv('TORCH_NPU_VERSION', ''))


def build_model_args(eval_config, tp_size):
    trust_remote_code = eval_config.get("trust_remote_code", False)
    max_model_len = eval_config.get("max_model_len", 4096)
    model_args = {
        "pretrained": eval_config["model_name"],
        "tensor_parallel_size": tp_size,
        "enforce_eager": True,
        "add_bos_token": True,
        "trust_remote_code": trust_remote_code,
        "max_model_len": max_model_len,
    }
    for s in ["max_images", "gpu_memory_utilization"]:
        val = eval_config.get(s, None)
        if val:
            model_args[s] = val
    return model_args


def generate_report(tp_size, eval_config, report_data, report_template,
                    report_output, env_config):
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(str(report_template))
    model_args = build_model_args(eval_config, tp_size)

    report_content = template.render(
        vllm_version=env_config.vllm_version,
        vllm_commit=env_config.vllm_commit,
        vllm_ascend_version=env_config.vllm_ascend_version,
        vllm_ascend_commit=env_config.vllm_ascend_commit,
        cann_version=env_config.cann_version,
        torch_version=env_config.torch_version,
        torch_npu_version=env_config.torch_npu_version,
        model_name=eval_config["model_name"],
        model_args=f"'{','.join(f'{k}={v}' for k, v in model_args.items())}'",
        model_type=eval_config.get("model", "vllm"),
        datasets=",".join([task["name"] for task in eval_config["tasks"]]),
        limit=eval_config.get("limit", None),
        batch_size="auto",
        num_fewshot=eval_config.get("num_fewshot", "N/A"),
        rows=report_data["rows"])

    os.makedirs(os.path.dirname(report_output), exist_ok=True)
    with open(report_output, 'w', encoding='utf-8') as f:
        f.write(report_content)


def test_lm_eval_correctness_param(config_filename, tp_size, report_template,
                                   report_output, env_config):
    eval_config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))
    model_args = build_model_args(eval_config, tp_size)
    eval_params = {
        "model": eval_config.get("model", "vllm"),
        "model_args": model_args,
        "tasks": [task["name"] for task in eval_config["tasks"]],
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "limit": eval_config.get("limit", None),
        "batch_size": "auto",
    }
    for s in ["num_fewshot"]:
        val = eval_config.get(s, None)
        if val:
            eval_params[s] = val

    results = lm_eval.simple_evaluate(**eval_params)
    success = True
    report_data: dict[str, list[dict]] = {"rows": []}

    for task in eval_config["tasks"]:
        for metric in task["metrics"]:
            ground_truth = metric["value"]
            measured_value = results["results"][task["name"]][metric["name"]]
            print(f"{task['name']} | {metric['name']}: "
                  f"ground_truth={ground_truth} | measured={measured_value}")
            success = success and bool(
                np.isclose(ground_truth, measured_value, rtol=RTOL))

            report_data["rows"].append({
                "task":
                task["name"],
                "metric":
                metric["name"],
                "value":
                f"✅{measured_value}" if success else f"❌{measured_value}",
                "stderr":
                results["results"][task["name"]][metric["name"].replace(
                    ',', '_stderr,', 1)]
            })
    generate_report(tp_size, eval_config, report_data, report_template,
                    report_output, env_config)
    assert success
