# {{ model_name }}

**vLLM Version**: vLLM: {{ vllm_version }} ([{{ vllm_commit[:7] }}](https://github.com/vllm-project/vllm/commit/{{ vllm_commit }}))
**vLLM Ascend Version**: {{ vllm_ascend_version }} ([{{ vllm_ascend_commit[:7] }}](https://github.com/vllm-project/vllm-ascend/commit/{{ vllm_ascend_commit }}))  
**Software Environment**: CANN: {{ cann_version }}, PyTorch: {{ torch_version }}, torch-npu: {{ torch_npu_version }}  
**Hardware Environment**: Atlas A2 Series  
**Datasets**: {{ datasets }}  
**Parallel Mode**: TP  
**Execution Mode**: ACLGraph  

**Command**:  
```bash
export MODEL_ARGS= {{ model_args }}
lm_eval --model {{ model_type }} --model_args $MODEL_ARGS --tasks {{ datasets }} \ 
--num_fewshot {{ num_fewshot }} --limit {{ limit }} --batch_size {{ batch_size}}
```

| Task                  | Filter           | n-shot | Metric      | Value     | Stderr |
|-----------------------|-----------------:|-------:|-------------|----------:|-------:|
{% for row in rows -%}
| {{ row.task.ljust(23) }} | {{ row.filter.rjust(15) }} | {{ row.n_shot | string.rjust(6) }} | {{ row.value }} | ± {{ "%.4f" | format(row.stderr | float) }} |
{% endfor %}