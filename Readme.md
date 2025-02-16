uv run examples/act_rebar/convert_aloha_data_to_lerobot.py --raw-dir dataset/insert7_plus --repo-id 1
uv run scripts/compute_norm_stats.py --config-name pi0_act_rebar
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_act_rebar --exp-name=act_rebar_finetuning_experiment --overwrite