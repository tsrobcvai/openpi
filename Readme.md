## Notes
* Action and state can be normalized to -1~1 or Z-normalization (based on [compute_norm_stats.py](scripts/compute_norm_stats.py) and xxx)
* Default image normalization is ...

## Commands
```
srun --partition=interactive --chdir=/n/fs/rebar/openpi --gres=gpu:1 --cpus-per-task=8 --mem=16G --pty bash
mkdir dataset
conda activate /n/fs/rebar/openpi/conda_env/pi0
export HF_DATASETS_CACHE=".cache/huggingface/datasets"
uv run examples/act_rebar/convert_aloha_data_to_lerobot.py --raw-dir dataset/insert7_small --repo-id insert7_small
uv run scripts/compute_norm_stats.py --config-name pi0_act_rebar_low_mem_finetune_relative
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_act_rebar_low_mem_finetune_relative --exp-name=pi0_rebar_lora_relative --overwrite
```
## Questions
* if we crop the last linear layer before output action / padding zero for action horizons before input, fine tuning 10 dim action will get a better performance?
* 