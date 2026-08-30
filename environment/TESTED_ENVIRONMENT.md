# Tested environment

- Python 3.9.23
- PyTorch 2.8.0+cu128
- CUDA runtime 12.8
- NVIDIA GeForce RTX 5090
- Linux under WSL2

Before assigning a public DOI, capture the complete 5090 environment with
`python -m pip freeze > environment/requirements-5090-lock.txt` and add the
output of `nvidia-smi` to `environment/nvidia-smi.txt`.
