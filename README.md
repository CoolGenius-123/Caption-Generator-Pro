# Caption Generator Pro

A simple Tkinter app for generating image captions with a LLaVA-style model.

## How to start

1. Double-click `install.bat` once to install everything from `requirements.txt`.
2. After that, double-click `startui.bat` to open the app.

## What to fill in

- **Model name**: keep the default unless you want to use a different Hugging Face model.
- **Hugging Face token (optional)**: fill this only if the model is private.
- **Image path**: choose one image for single-image mode.
- **Save path (optional)**: choose where to save the caption as `.txt` or `.json`.
- **Prompt**: edit this if you want a different caption style.
- **Output length**: lower values are faster; higher values produce longer captions.
- **Extra generation values**: use these only if you want to fine-tune the result.

## Single-image mode

1. Leave **Enable multi-image folder mode** unchecked.
2. Pick one image in **Image path**.
3. Optionally choose a **Save path**.
4. Click **Generate Caption**.

## Multi-image mode

1. Turn on **Enable multi-image folder mode**.
2. Choose the folder that contains your images.
3. Choose a folder where the captions should be saved.
4. Optionally set a **Filename prefix**.
5. Click **Generate Batch Captions**.

## Notes

- Shorter captions usually generate faster.
- The app supports common image formats like PNG, JPG, JPEG, WEBP, BMP, TIF, and TIFF.
- If you do not enter a save path, the caption will only show in the app.