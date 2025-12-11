# CIFAR-10 CNN

Skeleton repository for experimenting with a convolutional neural network on CIFAR-10 using PyTorch.

## Repository layout
- `config.py`: shared configuration (hyperparameters, paths, device selection).
- `data.py`: dataset/transforms and dataloaders.
- `model.py`: CNN architecture.
- `train.py`: training/evaluation loop entrypoint.
- `inference.py`: checkpoint loading, single-image prediction, or split evaluation.
- `app.py`: Streamlit UI to upload an image and view predictions.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio matplotlib tqdm
```
Install dependencies: `pip install -r requirements.txt`

## Usage
- Train (saves best checkpoint to `checkpoints/cifar10_cnn.pt`):  
  `python train.py`
- Evaluate a trained checkpoint on the test split:  
  `python inference.py --eval-split test`
- Classify a single image (resized to 32x32):  
  `python inference.py --image /path/to/image.jpg`
Pass `--checkpoint` to point at a custom checkpoint path.
- Run the Streamlit UI (after training so the checkpoint exists):  
  ```bash
  pip install streamlit pillow
  streamlit run app.py
  ```
  Upload an image and the app will show the top prediction plus a probability bar chart.

## Next steps
- Add richer logging (TensorBoard/Weights & Biases) and learning-rate scheduling.
- Include quick smoke tests to verify dataloaders, forward pass shapes, and a short training step.
