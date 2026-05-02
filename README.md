# Qwen Caption Studio

> A local Tkinter desktop app for generating prompt-ready image captions with a Qwen3.5 2B 4-bit vision-language model.

Target model:

```text
techwithsergiu/Qwen3.5-2B-bnb-4bit
```

## Preview

![Caption Generator Pro Screenshot](assets/Caption_Image_Generator.png)

## Features

- Single-image caption generation
- Folder batch captioning with stop-after-current-image behavior
- Qwen chat-template based vision-language inference
- Enhanced text-to-image prompt defaults with editable system and user prompts
- Drag-and-drop image loading with Browse fallback
- Speed presets, model prewarm, and FlashAttention-first loading with SDPA fallback
- Optional thinking mode control
- Optional image resizing before inference for lower GPU memory usage
- Save captions as `.txt` or `.json`
- Batch resume behavior by skipping output files that already exist
- Live RAM, CPU, GPU, temperature, timer, and progress status
- Background worker threads to keep the UI responsive

## Recommended Setup

Create a virtual environment first. Install PyTorch with CUDA from the official PyTorch selector for your GPU and driver, then install the app dependencies.

### Windows CMD

```cmd
cd "<repo path>\Caption-Generator-Pro"
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python code\caption_generator.py
```

### Windows PowerShell

```powershell
cd "<repo path>\Caption-Generator-Pro"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python code\caption_generator.py
```

## Quick Start

After setup, run `startui.bat` or launch the app directly:

```cmd
python code\caption_generator.py
```

## Performance Options

- **Speed preset** controls token budget and image resize limits for quality/speed tradeoffs.
- **Attention backend** defaults to FlashAttention 2 when compatible and falls back to SDPA if unavailable.
- **Prewarm after model load** runs a tiny warmup pass to reduce first-generation latency.

## Usage

### Single-image mode

1. Leave **Batch mode** unchecked.
2. Drop an image onto the app or use **Browse** to pick one image file.
3. Edit the system or user prompt if needed.
4. Optionally choose a save path.
5. Click **Generate**.

### Batch mode

1. Turn on **Batch mode**.
2. Choose the folder that contains your images.
3. Choose a folder where captions should be saved.
4. Optionally set a filename prefix.
5. Click **Generate**.

Batch jobs skip image outputs that already exist when **Skip existing output files** is enabled.

## Notes

- The first run may download model files from Hugging Face.
- Use a Hugging Face token in the UI if the model or cache access requires it.
- On memory-limited GPUs, keep image resizing enabled and close other GPU-heavy apps before loading the model.
- The app supports PNG, JPG, JPEG, WEBP, BMP, TIF, and TIFF images.
