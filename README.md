# CIFAR-10 CNN

Skeleton repository for experimenting with a convolutional neural network on CIFAR-10 using PyTorch.

## Repository layout
- `data/config.py`: shared configuration (hyperparameters, paths, device selection).
- `data/data.py`: dataset/transforms and dataloaders.
- `data/model.py`: CNN architecture.
- `data/train.py`: training/evaluation loop entrypoint.
- `data/inference.py`: checkpoint loading and single-image prediction entrypoint.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio matplotlib tqdm
```
Add a `requirements.txt` when the code stabilizes so installs can be pinned.

## Usage (once scripts are implemented)
- Train: `python data/train.py`
- Inference: `python data/inference.py --checkpoint <path-to-ckpt> --image <path-to-image>`
Adjust CLI arguments inside the scripts as you add `argparse` options.

## Next steps
- Flesh out the five stub modules (model, data, config, train, inference).
- Add a requirements file and a simple metrics/logging utility.
- Include basic tests or smoke scripts to verify training and inference flows.
