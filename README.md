# Stable Diffusion API

A simple API to work with Stable Diffusion models.

## Setup

Prerequisites:

- Python (3.10)
- PyTorch (2.0)

After you installed Python, install PyTorch by running:

```sh
pip install torch torchvision torchaudio
```

If you have a CUDA-capable GPU, install PyTorch with CUDA:

```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then, you install the dependencies:

```sh
pip install -r requirements.txt
```

To run the API, run:

```sh
cd sd-api
uvicorn api:app
```
