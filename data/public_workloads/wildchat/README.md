# WildChat workload

The paper uses a derived WildChat request-shape trace. Obtain WildChat directly from its upstream dataset page and follow its ODC-BY terms:

- Dataset: https://huggingface.co/datasets/allenai/WildChat
- Paper: https://arxiv.org/abs/2405.01470
- License: ODC-BY, according to the dataset card

Do not commit the raw conversation dataset. Use the repository's extraction utilities to create a derived trace under `data/workload_traces/`.
