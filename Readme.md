## Notes
* Action and state can be normalized to -1~1 or Z-normalization (based on [compute_norm_stats.py](scripts/compute_norm_stats.py) and xxx)
* Default image normalization is ...

## Commands
```
uv run examples/act_rebar/convert_aloha_data_to_lerobot.py --raw-dir dataset/insert7_plus --repo-id 1
uv run scripts/compute_norm_stats.py --config-name pi0_act_rebar_low_mem_finetune
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_act_rebar_low_mem_finetune --exp-name=act_rebar_finetuning_experiment --overwrite
```
## Questions
* if we crop the last linear layer before output action / padding zero for action horizons before input, fine tuning 10 dim action will get a better performance?
* 