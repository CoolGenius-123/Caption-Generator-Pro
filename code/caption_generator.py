from __future__ import annotations

import gc
import importlib.util
import json
import os
import re
import subprocess
import threading
import time
import traceback
from urllib.parse import unquote, urlparse
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception:
    DND_FILES = None
    TkinterDnD = None

try:
    import psutil
except Exception:
    psutil = None

import torch
from PIL import Image, ImageTk

from transformers import (
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Optional but strongly recommended for Qwen VL workflows
try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None


# -------- Model defaults --------

DEFAULT_MODEL_NAME = "techwithsergiu/Qwen3.5-2B-bnb-4bit"
DEFAULT_ATTENTION_BACKEND = "flash_attention_2"
DEFAULT_SPEED_PRESET = "Quality"

SPEED_PRESETS = {
    "Quality": {"max_new_tokens": 192, "max_image_side": 768, "thinking": False},
    "Fast": {"max_new_tokens": 144, "max_image_side": 704, "thinking": False},
    "Max Speed": {"max_new_tokens": 96, "max_image_side": 640, "thinking": False},
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise visual prompt writer for text-to-image models. "
    "Convert the image into one rich, grounded, prompt-ready description. "
    "Describe only what is visible or strongly implied by visual evidence. "
    "Use concrete nouns, visual adjectives, and production-friendly phrasing. "
    "Prioritize the main subject, composition, pose or action, setting, lighting, "
    "color palette, materials, textures, mood, camera angle or viewpoint, depth, "
    "and notable background details. Mention medium or style terms only when they "
    "are visually evident, such as anime illustration, studio photo, watercolor, "
    "3D render, cinematic lighting, macro shot, or flat graphic design. "
    "Do not guess identities, names, exact locations, hidden intent, artist names, "
    "copyrighted character names, or facts not shown in the image. "
    "Do not add commentary about the image or the task."
)

DEFAULT_USER_PROMPT = (
    "Write exactly one detailed natural-language prompt for a text-to-image model "
    "based on this image. Aim for 80 to 140 words. Start with the main subject "
    "and framing, then describe visible pose or action, facial expression if present, "
    "clothing or objects, environment, lighting, colors, materials, textures, mood, "
    "perspective, and background details. Make it vivid and useful for image generation "
    "while staying faithful to the image. Return only the prompt text, with no title, "
    "bullets, JSON, labels, explanations, or negative prompt."
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

APP_TITLE = "Qwen Caption Studio"


# -------- Utility helpers --------

def strip_thinking_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", name.strip())
    cleaned = cleaned.replace(" ", "_")
    return cleaned.strip("._ ") or "caption"


def format_seconds(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def resize_image_for_vram(image: Image.Image, max_side: int) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    scale = max_side / float(max(w, h))
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return image.resize(new_size, Image.LANCZOS)


def pick_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def get_cpu_ram_text() -> str:
    if psutil is None:
        return "RAM: N/A"
    vm = psutil.virtual_memory()
    return f"RAM: {vm.percent:.0f}%"


def get_cpu_text() -> str:
    if psutil is None:
        return "CPU: N/A"
    return f"CPU: {psutil.cpu_percent(interval=None):.0f}%"


def get_cpu_temp_text() -> str:
    if psutil is None:
        return "CPU TEMP: N/A"
    sensors_temperatures = getattr(psutil, "sensors_temperatures", None)
    if callable(sensors_temperatures):
        try:
            temps = sensors_temperatures(fahrenheit=False)
            for entries in temps.values():
                for entry in entries:
                    if getattr(entry, "current", None) is not None:
                        return f"CPU TEMP: {entry.current:.0f}°C"
        except Exception:
            pass
    return "CPU TEMP: N/A"


def run_nvidia_smi(query: str) -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        value = result.stdout.strip().splitlines()[0].strip()
        return value
    except Exception:
        return None


def get_gpu_text() -> str:
    if not torch.cuda.is_available():
        return "GPU: N/A"
    util = run_nvidia_smi("utilization.gpu")
    if util:
        return f"GPU: {util}%"
    try:
        return f"GPU: {torch.cuda.get_device_name(0)}"
    except Exception:
        return "GPU: CUDA"


def get_gpu_temp_text() -> str:
    if not torch.cuda.is_available():
        return "GPU TEMP: N/A"
    temp = run_nvidia_smi("temperature.gpu")
    return f"GPU TEMP: {temp}°C" if temp else "GPU TEMP: N/A"


def get_model_loader_class():
    """
    Try the most direct class first.
    Fall back to auto classes if the exact class is unavailable.
    """
    try:
        from transformers import Qwen3_5ForConditionalGeneration
        return Qwen3_5ForConditionalGeneration
    except Exception:
        pass

    try:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText
    except Exception:
        pass

    try:
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq
    except Exception:
        pass

    try:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM
    except Exception:
        pass

    raise ImportError(
        "Could not find a suitable Transformers model loader for Qwen3.5 multimodal. "
        "Please upgrade transformers."
    )


class StopGenerationCriteria(StoppingCriteria):
    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event
        super().__init__()

    def __call__(self, input_ids, scores, **kwargs):
        return torch.tensor(
            [self.stop_event.is_set()],
            device=input_ids.device,
            dtype=torch.bool,
        )


# -------- Main App --------

class QwenCaptionStudio:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1360x900")
        self.root.minsize(1180, 820)
        self.root.configure(bg="#0b1220")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.worker: threading.Thread | None = None
        self.stop_requested = threading.Event()

        self.processor: Any | None = None
        self.model: Any | None = None
        self.loaded_model_name: str | None = None
        self.loaded_device: str | None = None
        self.loaded_dtype: torch.dtype | None = None
        self.loaded_attn_backend: str | None = None

        self.preview_photo: ImageTk.PhotoImage | None = None
        self.image_entry: tk.Entry | None = None
        self.timer_running = False
        self.timer_start = 0.0
        self.timer_job: str | None = None
        self.last_elapsed = "--:--:--"

        self._make_variables()
        self._build_styles()
        self._build_ui()
        self._setup_drag_drop()
        self._update_system_info()
        self._set_status("Ready. Load the model or pick an image and generate a caption.")

    # ---------- Variables ----------

    def _make_variables(self):
        self.model_name_var = tk.StringVar(value=DEFAULT_MODEL_NAME)
        self.hf_token_var = tk.StringVar(value="")
        self.image_path_var = tk.StringVar(value="")
        self.save_path_var = tk.StringVar(value="")
        self.save_format_var = tk.StringVar(value="txt")

        self.batch_mode_var = tk.BooleanVar(value=False)
        self.batch_folder_var = tk.StringVar(value="")
        self.batch_save_folder_var = tk.StringVar(value="")
        self.batch_prefix_var = tk.StringVar(value="")
        self.skip_existing_var = tk.BooleanVar(value=True)

        self.speed_preset_var = tk.StringVar(value=DEFAULT_SPEED_PRESET)
        self.prewarm_var = tk.BooleanVar(value=True)
        self.enable_thinking_var = tk.BooleanVar(value=False)
        self.max_new_tokens_var = tk.IntVar(value=SPEED_PRESETS[DEFAULT_SPEED_PRESET]["max_new_tokens"])
        self.max_image_side_var = tk.IntVar(value=SPEED_PRESETS[DEFAULT_SPEED_PRESET]["max_image_side"])

        self.do_sample_var = tk.BooleanVar(value=False)
        self.temperature_var = tk.DoubleVar(value=0.3)
        self.top_p_var = tk.DoubleVar(value=0.9)
        self.top_k_var = tk.IntVar(value=40)
        self.repetition_penalty_var = tk.DoubleVar(value=1.05)
        self.attn_backend_var = tk.StringVar(value=DEFAULT_ATTENTION_BACKEND)

        self.cpu_var = tk.StringVar(value="CPU: --")
        self.ram_var = tk.StringVar(value="RAM: --")
        self.gpu_var = tk.StringVar(value="GPU: --")
        self.cpu_temp_var = tk.StringVar(value="CPU TEMP: --")
        self.gpu_temp_var = tk.StringVar(value="GPU TEMP: --")
        self.timer_var = tk.StringVar(value="Elapsed: --:--:--")
        self.current_item_var = tk.StringVar(value="Current: none")
        self.model_state_var = tk.StringVar(value="Model: not loaded")
        self.batch_progress_var = tk.StringVar(value="")
        self.prompt_detail_var = tk.StringVar(value="")

    # ---------- UI ----------

    def _build_styles(self):
        self.bg = "#0b1220"
        self.panel = "#101a2d"
        self.panel_alt = "#13213a"
        self.card = "#16243f"
        self.border = "#253655"
        self.text = "#e5edf8"
        self.muted = "#9fb1ca"
        self.accent = "#59c3ff"
        self.good = "#34d399"
        self.warn = "#fbbf24"
        self.bad = "#f87171"

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=self.bg)
        style.configure("Card.TFrame", background=self.panel)
        style.configure("TLabel", background=self.bg, foreground=self.text, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=self.bg, foreground=self.muted, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=self.bg, foreground=self.text, font=("Segoe UI", 22, "bold"))
        style.configure("Section.TLabel", background=self.panel, foreground=self.text, font=("Segoe UI", 11, "bold"))
        style.configure("TNotebook", background=self.bg, borderwidth=0)
        style.configure("TNotebook.Tab", font=("Segoe UI", 10), padding=(10, 6))
        style.configure("TCombobox", padding=5)
        style.configure("TCheckbutton", background=self.panel, foreground=self.text)

        style.configure(
            "Primary.TButton",
            background=self.accent,
            foreground="#08121f",
            font=("Segoe UI", 10, "bold"),
            padding=(12, 8),
        )
        style.map("Primary.TButton", background=[("active", "#8bd8ff"), ("disabled", "#36546e")])

        style.configure(
            "Soft.TButton",
            background="#253655",
            foreground=self.text,
            font=("Segoe UI", 10),
            padding=(10, 8),
        )
        style.map("Soft.TButton", background=[("active", "#34486d")])

        style.configure(
            "Danger.TButton",
            background="#dc2626",
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padding=(10, 8),
        )
        style.map("Danger.TButton", background=[("active", "#ef4444"), ("disabled", "#692020")])

    def _build_ui(self):
        self._build_header()

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        left = tk.Frame(body, bg=self.bg, width=460)
        right = tk.Frame(body, bg=self.bg)

        body.add(left, weight=0)
        body.add(right, weight=1)

        self._build_left_panel(left)
        self._build_right_panel(right)

    def _build_header(self):
        header = tk.Frame(self.root, bg=self.bg)
        header.pack(fill="x", padx=16, pady=(14, 10))

        tk.Label(
            header,
            text=APP_TITLE,
            bg=self.bg,
            fg=self.text,
            font=("Segoe UI", 22, "bold"),
        ).pack(anchor="w")

        tk.Label(
            header,
            text="Clean local image captioning with Qwen3.5 2B 4-bit, system prompts, thinking control, and batch export.",
            bg=self.bg,
            fg=self.muted,
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(2, 10))

        chips = tk.Frame(header, bg=self.bg)
        chips.pack(fill="x")

        for var in [self.cpu_var, self.ram_var, self.gpu_var, self.cpu_temp_var, self.gpu_temp_var, self.timer_var]:
            self._make_chip(chips, var)

    def _make_chip(self, parent: tk.Widget, var: tk.StringVar):
        frame = tk.Frame(parent, bg=self.card, highlightbackground=self.border, highlightthickness=1)
        frame.pack(side="left", padx=(0, 8))
        tk.Label(
            frame,
            textvariable=var,
            bg=self.card,
            fg=self.text,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=6,
        ).pack()

    def _build_left_panel(self, parent: tk.Widget):
        wrapper = tk.Frame(parent, bg=self.panel, highlightbackground=self.border, highlightthickness=1)
        wrapper.pack(fill="both", expand=True)

        notebook = ttk.Notebook(wrapper)
        notebook.pack(fill="both", expand=True, padx=10, pady=(10, 0))

        tab_run_outer = tk.Frame(notebook, bg=self.panel)
        tab_prompts_outer = tk.Frame(notebook, bg=self.panel)
        tab_batch_outer = tk.Frame(notebook, bg=self.panel)
        tab_advanced_outer = tk.Frame(notebook, bg=self.panel)

        self.tab_run = self._make_scrollable_tab(tab_run_outer)
        self.tab_prompts = self._make_scrollable_tab(tab_prompts_outer)
        self.tab_batch = self._make_scrollable_tab(tab_batch_outer)
        self.tab_advanced = self._make_scrollable_tab(tab_advanced_outer)

        notebook.add(tab_run_outer, text="Run")
        notebook.add(tab_prompts_outer, text="Prompts")
        notebook.add(tab_batch_outer, text="Batch")
        notebook.add(tab_advanced_outer, text="Advanced")

        self._build_run_tab()
        self._build_prompts_tab()
        self._build_batch_tab()
        self._build_advanced_tab()
        self._build_utility_bar(wrapper)

    def _make_scrollable_tab(self, parent: tk.Widget) -> tk.Frame:
        canvas = tk.Canvas(parent, bg=self.panel, bd=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=self.panel)

        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def refresh_scrollregion(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def sync_width(event):
            canvas.itemconfigure(window_id, width=event.width)

        inner.bind("<Configure>", refresh_scrollregion)
        canvas.bind("<Configure>", sync_width)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return inner

    def _build_utility_bar(self, parent: tk.Widget):
        bar = tk.Frame(parent, bg=self.panel, highlightbackground=self.border, highlightthickness=1)
        bar.pack(fill="x", padx=10, pady=10)

        utility_row = tk.Frame(bar, bg=self.panel)
        utility_row.pack(fill="x", padx=8, pady=8)
        ttk.Button(utility_row, text="Clear Output", style="Soft.TButton", command=self.clear_output).pack(side="left", fill="x", expand=True)
        ttk.Button(utility_row, text="Reset Defaults", style="Soft.TButton", command=self.reset_defaults).pack(side="left", fill="x", expand=True, padx=(8, 0))

    def _build_right_panel(self, parent: tk.Widget):
        top = tk.Frame(parent, bg=self.bg)
        top.pack(fill="x", pady=(0, 10))

        state_row = tk.Frame(top, bg=self.bg)
        state_row.pack(fill="x")

        tk.Label(
            state_row,
            textvariable=self.model_state_var,
            bg=self.bg,
            fg=self.accent,
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left")

        tk.Label(
            state_row,
            textvariable=self.current_item_var,
            bg=self.bg,
            fg=self.muted,
            font=("Segoe UI", 10),
        ).pack(side="right")

        preview_card = tk.Frame(parent, bg=self.panel, highlightbackground=self.border, highlightthickness=1)
        preview_card.pack(fill="x", pady=(0, 10))

        preview_header = tk.Frame(preview_card, bg=self.panel)
        preview_header.pack(fill="x", padx=12, pady=(10, 8))

        tk.Label(preview_header, text="Preview", bg=self.panel, fg=self.text, font=("Segoe UI", 12, "bold")).pack(side="left")
        tk.Label(preview_header, textvariable=self.batch_progress_var, bg=self.panel, fg=self.muted, font=("Segoe UI", 9)).pack(side="right")

        self.preview_canvas = tk.Canvas(
            preview_card,
            width=760,
            height=320,
            bg="#0a1322",
            bd=0,
            highlightbackground=self.border,
            highlightthickness=1,
        )
        self.preview_canvas.pack(fill="x", padx=12, pady=(0, 12))
        self._clear_preview()

        output_card = tk.Frame(parent, bg=self.panel, highlightbackground=self.border, highlightthickness=1)
        output_card.pack(fill="both", expand=True)

        output_tabs = ttk.Notebook(output_card)
        output_tabs.pack(fill="both", expand=True, padx=10, pady=10)

        out_tab = tk.Frame(output_tabs, bg=self.panel)
        log_tab = tk.Frame(output_tabs, bg=self.panel)

        output_tabs.add(out_tab, text="Caption Output")
        output_tabs.add(log_tab, text="Debug Logs")

        self.output_text = scrolledtext.ScrolledText(
            out_tab,
            wrap=tk.WORD,
            bg="#0a1322",
            fg=self.text,
            insertbackground=self.text,
            relief="flat",
            font=("Segoe UI", 11),
        )
        self.output_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.output_text.insert("1.0", "Ready. Generate a caption to see results.")
        self.output_text.configure(state="disabled")

        self.debug_text = scrolledtext.ScrolledText(
            log_tab,
            wrap=tk.WORD,
            bg="#08101c",
            fg="#9ad0ff",
            insertbackground=self.text,
            relief="flat",
            font=("Consolas", 9),
        )
        self.debug_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.debug_text.configure(state="disabled")

        status_bar = tk.Frame(parent, bg=self.bg)
        status_bar.pack(fill="x", pady=(8, 0))
        self.status_label = tk.Label(status_bar, text="", bg=self.bg, fg=self.muted, anchor="w", font=("Segoe UI", 10))
        self.status_label.pack(fill="x")

    def _build_run_tab(self):
        frame = self.tab_run

        self._label(frame, "Model")
        self._entry(frame, self.model_name_var)

        self._label(frame, "Hugging Face token (optional)")
        token_entry = tk.Entry(
            frame,
            textvariable=self.hf_token_var,
            bg="#0d1628",
            fg=self.text,
            insertbackground=self.text,
            relief="flat",
            font=("Segoe UI", 10),
        )
        token_entry.pack(fill="x", padx=14, pady=(4, 10), ipady=7)

        row1 = tk.Frame(frame, bg=self.panel)
        row1.pack(fill="x", padx=14, pady=(0, 10))
        ttk.Button(row1, text="Load Model", style="Soft.TButton", command=self.load_model_async).pack(side="left", fill="x", expand=True)
        ttk.Button(row1, text="Unload Model", style="Soft.TButton", command=self.unload_model).pack(side="left", fill="x", expand=True, padx=(8, 0))

        mode_card = tk.Frame(frame, bg=self.panel_alt, highlightbackground=self.border, highlightthickness=1)
        mode_card.pack(fill="x", padx=14, pady=(0, 10))
        tk.Checkbutton(
            mode_card,
            text="Enable thinking mode",
            variable=self.enable_thinking_var,
            bg=self.panel_alt,
            fg=self.text,
            activebackground=self.panel_alt,
            activeforeground=self.text,
            selectcolor=self.card,
            relief="flat",
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=12, pady=10)

        self._build_performance_card(frame)

        run_row = tk.Frame(frame, bg=self.panel)
        run_row.pack(fill="x", padx=14, pady=(0, 10))
        self.generate_button = ttk.Button(run_row, text="Generate", style="Primary.TButton", command=self.generate_async)
        self.generate_button.pack(side="left", fill="x", expand=True)

        self.stop_button = ttk.Button(run_row, text="Stop", style="Danger.TButton", command=self.request_stop, state="disabled")
        self.stop_button.pack(side="left", fill="x", expand=True, padx=(8, 0))

        self._label(frame, "Image")
        self.image_entry = self._path_row(frame, self.image_path_var, self.choose_image)

        self._label(frame, "Save path (optional for single image)")
        self._path_row(frame, self.save_path_var, self.choose_save_path)

        combo_row = tk.Frame(frame, bg=self.panel)
        combo_row.pack(fill="x", padx=14, pady=(0, 10))
        tk.Label(combo_row, text="Save format", bg=self.panel, fg=self.muted, font=("Segoe UI", 10)).pack(side="left")
        ttk.Combobox(
            combo_row,
            textvariable=self.save_format_var,
            values=("txt", "json"),
            state="readonly",
            width=10,
        ).pack(side="right")

        self._build_prompt_detail_card(frame)

    def _build_performance_card(self, parent: tk.Widget):
        card = tk.Frame(parent, bg=self.panel_alt, highlightbackground=self.border, highlightthickness=1)
        card.pack(fill="x", padx=14, pady=(0, 10))
        self._label_inside(card, "Performance")

        self._label_inside_small(card, "Speed preset")
        preset_combo = ttk.Combobox(
            card,
            textvariable=self.speed_preset_var,
            values=tuple(SPEED_PRESETS.keys()),
            state="readonly",
        )
        preset_combo.pack(fill="x", padx=12, pady=(0, 8))
        preset_combo.bind("<<ComboboxSelected>>", lambda _event: self._apply_speed_preset())

        self._label_inside_small(card, "Attention backend")
        ttk.Combobox(
            card,
            textvariable=self.attn_backend_var,
            values=("flash_attention_2", "sdpa", "eager"),
            state="readonly",
        ).pack(fill="x", padx=12, pady=(0, 8))

        tk.Checkbutton(
            card,
            text="Prewarm after model load",
            variable=self.prewarm_var,
            bg=self.panel_alt,
            fg=self.text,
            activebackground=self.panel_alt,
            activeforeground=self.text,
            selectcolor=self.card,
            relief="flat",
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=12, pady=(0, 6))

        tk.Label(
            card,
            text="FlashAttention is tried first and falls back to SDPA if unavailable.",
            bg=self.panel_alt,
            fg=self.muted,
            font=("Segoe UI", 9),
            justify="left",
            wraplength=360,
        ).pack(anchor="w", padx=12, pady=(0, 10))

    def _build_prompt_detail_card(self, parent: tk.Widget):
        settings_card = tk.Frame(parent, bg=self.panel_alt, highlightbackground=self.border, highlightthickness=1)
        settings_card.pack(fill="x", padx=14, pady=(0, 10))

        self._label_inside(settings_card, "Prompt detail")
        tk.Label(
            settings_card,
            textvariable=self.prompt_detail_var,
            bg=self.panel_alt,
            fg=self.muted,
            font=("Segoe UI", 9),
            justify="left",
            wraplength=360,
        ).pack(anchor="w", padx=12, pady=(0, 8))

        self._scale_block(settings_card, "Max new tokens", self.max_new_tokens_var, 32, 384, 8)
        self._scale_block(settings_card, "Max image side", self.max_image_side_var, 512, 1024, 64)
        self._update_prompt_detail()

    def _apply_speed_preset(self):
        preset_name = self.speed_preset_var.get()
        preset = SPEED_PRESETS.get(preset_name, SPEED_PRESETS[DEFAULT_SPEED_PRESET])
        self.max_new_tokens_var.set(int(preset["max_new_tokens"]))
        self.max_image_side_var.set(int(preset["max_image_side"]))
        self.enable_thinking_var.set(bool(preset["thinking"]))
        self._update_prompt_detail()
        self._set_status(f"Speed preset applied: {preset_name}.")

    def _update_prompt_detail(self):
        if not hasattr(self, "prompt_detail_var"):
            return
        self.prompt_detail_var.set(
            f"{self.speed_preset_var.get()} preset: "
            f"{int(self.max_new_tokens_var.get())} tokens, "
            f"{int(self.max_image_side_var.get())} px image side."
        )

    def _build_prompts_tab(self):
        frame = self.tab_prompts

        self._label(frame, "System prompt")
        self.system_prompt_text = scrolledtext.ScrolledText(
            frame,
            height=8,
            wrap=tk.WORD,
            bg="#0d1628",
            fg=self.text,
            insertbackground=self.text,
            relief="flat",
            font=("Segoe UI", 10),
        )
        self.system_prompt_text.pack(fill="x", padx=14, pady=(4, 12))
        self.system_prompt_text.insert("1.0", DEFAULT_SYSTEM_PROMPT)

        self._label(frame, "User prompt")
        self.user_prompt_text = scrolledtext.ScrolledText(
            frame,
            height=12,
            wrap=tk.WORD,
            bg="#0d1628",
            fg=self.text,
            insertbackground=self.text,
            relief="flat",
            font=("Segoe UI", 10),
        )
        self.user_prompt_text.pack(fill="both", expand=True, padx=14, pady=(4, 12))
        self.user_prompt_text.insert("1.0", DEFAULT_USER_PROMPT)

    def _build_batch_tab(self):
        frame = self.tab_batch

        top = tk.Frame(frame, bg=self.panel)
        top.pack(fill="x", padx=14, pady=(12, 10))

        tk.Checkbutton(
            top,
            text="Enable batch folder mode",
            variable=self.batch_mode_var,
            bg=self.panel,
            fg=self.text,
            activebackground=self.panel,
            activeforeground=self.text,
            selectcolor=self.card,
            relief="flat",
            font=("Segoe UI", 10),
        ).pack(anchor="w")

        self._label(frame, "Batch image folder")
        self._path_row(frame, self.batch_folder_var, self.choose_batch_folder)

        self._label(frame, "Batch save folder")
        self._path_row(frame, self.batch_save_folder_var, self.choose_batch_save_folder)

        self._label(frame, "Optional filename prefix")
        self._entry(frame, self.batch_prefix_var)

        opts = tk.Frame(frame, bg=self.panel)
        opts.pack(fill="x", padx=14, pady=(6, 0))
        tk.Checkbutton(
            opts,
            text="Skip existing output files",
            variable=self.skip_existing_var,
            bg=self.panel,
            fg=self.text,
            activebackground=self.panel,
            activeforeground=self.text,
            selectcolor=self.card,
            relief="flat",
            font=("Segoe UI", 10),
        ).pack(anchor="w")

        note = tk.Label(
            frame,
            text="Batch mode processes one image at a time to keep memory use predictable.",
            bg=self.panel,
            fg=self.muted,
            font=("Segoe UI", 9),
            justify="left",
            wraplength=380,
        )
        note.pack(anchor="w", padx=14, pady=(10, 0))

    def _build_advanced_tab(self):
        frame = self.tab_advanced

        self._label(frame, "Attention backend")
        ttk.Combobox(
            frame,
            textvariable=self.attn_backend_var,
            values=("flash_attention_2", "sdpa", "eager"),
            state="readonly",
        ).pack(fill="x", padx=14, pady=(4, 10))

        adv_card = tk.Frame(frame, bg=self.panel_alt, highlightbackground=self.border, highlightthickness=1)
        adv_card.pack(fill="x", padx=14, pady=(0, 10))

        self._label_inside(adv_card, "Sampling")
        tk.Checkbutton(
            adv_card,
            text="Use sampling",
            variable=self.do_sample_var,
            bg=self.panel_alt,
            fg=self.text,
            activebackground=self.panel_alt,
            activeforeground=self.text,
            selectcolor=self.card,
            relief="flat",
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=12, pady=(0, 6))

        self._scale_block(adv_card, "Temperature", self.temperature_var, 0.1, 1.2, 0.05)
        self._scale_block(adv_card, "Top-p", self.top_p_var, 0.1, 1.0, 0.05)
        self._scale_block(adv_card, "Top-k", self.top_k_var, 0, 100, 1)
        self._scale_block(adv_card, "Repetition penalty", self.repetition_penalty_var, 1.0, 1.3, 0.01)

        help_label = tk.Label(
            frame,
            text=(
                "Recommended defaults for memory-constrained CUDA systems:\n"
                "• flash_attention_2 first, SDPA fallback\n"
                "• thinking off for normal captioning\n"
                "• max image side 768\n"
                "• max new tokens 144 to 192\n"
                "• batch size 1"
            ),
            bg=self.panel,
            fg=self.muted,
            font=("Segoe UI", 9),
            justify="left",
        )
        help_label.pack(anchor="w", padx=14, pady=(4, 0))

    def _label(self, parent: tk.Widget, text: str):
        tk.Label(parent, text=text, bg=self.panel, fg=self.text, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=14, pady=(12, 0))

    def _label_inside(self, parent: tk.Widget, text: str):
        tk.Label(parent, text=text, bg=self.panel_alt, fg=self.text, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 8))

    def _label_inside_small(self, parent: tk.Widget, text: str):
        tk.Label(parent, text=text, bg=self.panel_alt, fg=self.muted, font=("Segoe UI", 9)).pack(anchor="w", padx=12, pady=(0, 4))

    def _entry(self, parent: tk.Widget, variable: tk.StringVar):
        entry = tk.Entry(
            parent,
            textvariable=variable,
            bg="#0d1628",
            fg=self.text,
            insertbackground=self.text,
            relief="flat",
            font=("Segoe UI", 10),
        )
        entry.pack(fill="x", padx=14, pady=(4, 10), ipady=7)
        return entry

    def _path_row(self, parent: tk.Widget, variable: tk.StringVar, browse_cmd):
        row = tk.Frame(parent, bg=self.panel)
        row.pack(fill="x", padx=14, pady=(4, 10))
        entry = tk.Entry(
            row,
            textvariable=variable,
            bg="#0d1628",
            fg=self.text,
            insertbackground=self.text,
            relief="flat",
            font=("Segoe UI", 10),
        )
        entry.pack(side="left", fill="x", expand=True, ipady=7)
        tk.Button(
            row,
            text="Browse",
            command=browse_cmd,
            bg="#253655",
            fg=self.text,
            activebackground="#31476c",
            activeforeground=self.text,
            relief="flat",
            font=("Segoe UI", 10),
            padx=12,
            pady=6,
        ).pack(side="right", padx=(8, 0))
        return entry

    def _setup_drag_drop(self):
        if DND_FILES is None:
            self._log("Drag and drop disabled. Install tkinterdnd2 to enable it.")
            return

        registered = 0
        seen: set[str] = set()
        targets = [self.root, *self._iter_widgets(self.root)]

        for target in targets:
            widget_name = str(target)
            if widget_name in seen:
                continue
            seen.add(widget_name)
            try:
                target.drop_target_register(DND_FILES)
                target.dnd_bind("<<Drop>>", self._handle_image_drop)
                registered += 1
            except Exception as exc:
                self._log(f"Could not enable drag and drop for {target}: {exc}")

        if registered:
            self._log(f"Drag and drop enabled on {registered} UI target(s).")

    def _iter_widgets(self, widget: tk.Widget):
        for child in widget.winfo_children():
            yield child
            yield from self._iter_widgets(child)

    def _handle_image_drop(self, event):
        image_path = self._first_image_from_drop(event.data)
        if image_path is None:
            messagebox.showerror(
                "Unsupported file",
                "Drop a supported image file: PNG, JPG, JPEG, WEBP, BMP, TIF, or TIFF.",
            )
            return

        self._load_image_path(image_path, "drop")

    def _first_image_from_drop(self, data: str) -> Path | None:
        try:
            raw_paths = self.root.tk.splitlist(data)
        except tk.TclError:
            raw_paths = [data]

        for raw_path in raw_paths:
            image_path = self._normalize_dropped_path(raw_path)
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                return image_path
        return None

    def _normalize_dropped_path(self, raw_path: str) -> Path:
        value = raw_path.strip().strip("{}").strip()
        parsed = urlparse(value)
        if parsed.scheme == "file":
            value = unquote(parsed.path)
            if os.name == "nt" and re.match(r"^/[A-Za-z]:/", value):
                value = value[1:]
        return Path(value)

    def _scale_block(self, parent: tk.Widget, label: str, variable, from_, to, resolution):
        block = tk.Frame(parent, bg=parent.cget("bg"))
        block.pack(fill="x", padx=12, pady=(0, 8))
        top = tk.Frame(block, bg=parent.cget("bg"))
        top.pack(fill="x")
        value_var = tk.StringVar()
        value_var.set(str(variable.get()))
        tk.Label(top, text=label, bg=parent.cget("bg"), fg=self.text, font=("Segoe UI", 10)).pack(side="left")
        tk.Label(top, textvariable=value_var, bg=parent.cget("bg"), fg=self.accent, font=("Segoe UI", 10, "bold")).pack(side="right")

        def on_change(value):
            f = float(value)
            if float(resolution).is_integer():
                value_var.set(str(int(f)))
            else:
                value_var.set(f"{f:.2f}")
            self._update_prompt_detail()

        scale = tk.Scale(
            block,
            from_=from_,
            to=to,
            resolution=resolution,
            orient="horizontal",
            variable=variable,
            bg=parent.cget("bg"),
            fg=self.text,
            troughcolor="#253655",
            highlightthickness=0,
            activebackground=self.accent,
            relief="flat",
            command=on_change,
        )
        scale.pack(fill="x")
        on_change(variable.get())
        variable.trace_add("write", lambda *_args: on_change(variable.get()))

    # ---------- UI Actions ----------

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        if path:
            self._load_image_path(Path(path), "browse")

    def _load_image_path(self, image_path: Path, source: str):
        self.image_path_var.set(str(image_path))
        self._update_preview(str(image_path))
        self.current_item_var.set(f"Current: {image_path.name}")
        if source == "drop":
            self._set_status(f"Image loaded from drop: {image_path.name}", self.good)
        else:
            self._set_status(f"Image loaded: {image_path.name}", self.good)

    def choose_save_path(self):
        ext = ".json" if self.save_format_var.get() == "json" else ".txt"
        path = filedialog.asksaveasfilename(
            title="Choose save path",
            defaultextension=ext,
            filetypes=[("Text", "*.txt"), ("JSON", "*.json")],
        )
        if path:
            self.save_path_var.set(path)
            suffix = Path(path).suffix.lower().lstrip(".")
            if suffix in {"txt", "json"}:
                self.save_format_var.set(suffix)

    def choose_batch_folder(self):
        folder = filedialog.askdirectory(title="Choose image folder")
        if folder:
            self.batch_folder_var.set(folder)

    def choose_batch_save_folder(self):
        folder = filedialog.askdirectory(title="Choose batch save folder")
        if folder:
            self.batch_save_folder_var.set(folder)

    def clear_output(self):
        self._set_output("Ready. Generate a caption to see results.")
        self._set_status("Output cleared.")

    def reset_defaults(self):
        self.model_name_var.set(DEFAULT_MODEL_NAME)
        self.hf_token_var.set("")
        self.image_path_var.set("")
        self.save_path_var.set("")
        self.save_format_var.set("txt")
        self.batch_mode_var.set(False)
        self.batch_folder_var.set("")
        self.batch_save_folder_var.set("")
        self.batch_prefix_var.set("")
        self.skip_existing_var.set(True)

        self.speed_preset_var.set(DEFAULT_SPEED_PRESET)
        self.prewarm_var.set(True)
        self._apply_speed_preset()
        self.do_sample_var.set(False)
        self.temperature_var.set(0.3)
        self.top_p_var.set(0.9)
        self.top_k_var.set(40)
        self.repetition_penalty_var.set(1.05)
        self.attn_backend_var.set(DEFAULT_ATTENTION_BACKEND)

        self.system_prompt_text.delete("1.0", tk.END)
        self.system_prompt_text.insert("1.0", DEFAULT_SYSTEM_PROMPT)

        self.user_prompt_text.delete("1.0", tk.END)
        self.user_prompt_text.insert("1.0", DEFAULT_USER_PROMPT)

        self._clear_preview()
        self.current_item_var.set("Current: none")
        self.batch_progress_var.set("")
        self._set_status("Defaults restored.")

    # ---------- Output / Status / Logs ----------

    def _set_output(self, text: str):
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", text)
        self.output_text.configure(state="disabled")

    def _append_output(self, text: str):
        self.output_text.configure(state="normal")
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.output_text.configure(state="disabled")

    def _set_status(self, text: str, color: str | None = None):
        self.status_label.config(text=text, fg=color or self.muted)

    def _log(self, text: str):
        def inner():
            self.debug_text.configure(state="normal")
            self.debug_text.insert(tk.END, text.rstrip() + "\n")
            self.debug_text.see(tk.END)
            self.debug_text.configure(state="disabled")
        self.root.after(0, inner)

    def _clear_logs(self):
        self.debug_text.configure(state="normal")
        self.debug_text.delete("1.0", tk.END)
        self.debug_text.configure(state="disabled")

    # ---------- Preview ----------

    def _update_preview(self, image_path: str):
        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((760, 320))
            self.preview_photo = ImageTk.PhotoImage(image)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(380, 160, image=self.preview_photo)
        except Exception as exc:
            self._clear_preview(f"Preview failed\n{exc}")

    def _clear_preview(self, text: str = "Drop an image here or use Browse"):
        self.preview_photo = None
        self.preview_canvas.delete("all")
        self.preview_canvas.create_text(
            380,
            160,
            text=text,
            fill=self.muted,
            font=("Segoe UI", 12),
            justify="center",
        )

    # ---------- Monitoring / Timer ----------

    def _update_system_info(self):
        self.cpu_var.set(get_cpu_text())
        self.ram_var.set(get_cpu_ram_text())
        self.gpu_var.set(get_gpu_text())
        self.cpu_temp_var.set(get_cpu_temp_text())
        self.gpu_temp_var.set(get_gpu_temp_text())
        self.root.after(1000, self._update_system_info)

    def _start_timer(self):
        self.timer_running = True
        self.timer_start = time.perf_counter()
        self._update_timer()

    def _update_timer(self):
        if not self.timer_running:
            return
        elapsed = format_seconds(time.perf_counter() - self.timer_start)
        self.timer_var.set(f"Elapsed: {elapsed}")
        self.timer_job = self.root.after(250, self._update_timer)

    def _stop_timer(self):
        if self.timer_job:
            try:
                self.root.after_cancel(self.timer_job)
            except Exception:
                pass
            self.timer_job = None

        if self.timer_running:
            elapsed = time.perf_counter() - self.timer_start
            self.last_elapsed = format_seconds(elapsed)

        self.timer_running = False
        self.timer_var.set(f"Elapsed: {self.last_elapsed}")

    # ---------- Model Handling ----------

    def load_model_async(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "A job is already running.")
            return

        self.stop_requested.clear()
        self.worker = threading.Thread(target=self._worker_load_model, daemon=True)
        self.worker.start()

    def _worker_load_model(self):
        try:
            self.root.after(0, lambda: self._set_status("Loading model...", self.accent))
            self._load_model()
            self.root.after(0, lambda: self._set_status("Model loaded successfully.", self.good))
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            self._log(traceback.format_exc())
            self.root.after(0, lambda: self._set_status(err, self.bad))
            self.root.after(0, lambda: messagebox.showerror("Model load failed", err))
        finally:
            self.worker = None

    def _load_model(self):
        model_name = self.model_name_var.get().strip() or DEFAULT_MODEL_NAME
        hf_token = self.hf_token_var.get().strip()
        requested_attn_backend = self.attn_backend_var.get().strip() or DEFAULT_ATTENTION_BACKEND
        resolved_attn_backend = self._resolve_attention_backend(requested_attn_backend)

        if (
            self.model is not None
            and self.processor is not None
            and self.loaded_model_name == model_name
            and self.loaded_attn_backend == resolved_attn_backend
        ):
            self._log("Model already loaded. Reusing current model.")
            return

        self.unload_model(silent=True)

        torch_dtype = pick_torch_dtype()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loader_cls = get_model_loader_class()

        self._log(f"Loading model: {model_name}")
        self._log(f"Loader class: {loader_cls.__name__}")
        self._log(f"Device: {device}")
        self._log(f"Dtype: {torch_dtype}")
        self._log(f"Requested attention backend: {requested_attn_backend}")
        self._log(f"Resolved attention backend: {resolved_attn_backend}")
        self._log(f"Speed preset: {self.speed_preset_var.get()}")

        token_kw = {"token": hf_token} if hf_token else {}

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            **token_kw,
        )

        backend_attempts = [resolved_attn_backend]
        if requested_attn_backend == "flash_attention_2" and resolved_attn_backend == "flash_attention_2":
            backend_attempts.append("sdpa")

        last_error: Exception | None = None
        for backend in backend_attempts:
            try:
                load_kwargs = self._build_model_load_kwargs(backend, token_kw, torch_dtype, device)
                self._log(f"Trying attention backend: {backend}")
                self.model = loader_cls.from_pretrained(model_name, **load_kwargs)
                resolved_attn_backend = backend
                break
            except Exception as exc:
                last_error = exc
                if backend == "flash_attention_2" and "sdpa" in backend_attempts:
                    self._log(f"FlashAttention load failed; retrying with SDPA. {type(exc).__name__}: {exc}")
                    self.model = None
                    gc.collect()
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    continue
                raise

        if self.model is None:
            raise RuntimeError(f"Model load failed: {last_error}")

        self.model.eval()

        self.loaded_model_name = model_name
        self.loaded_device = device
        self.loaded_dtype = torch_dtype
        self.loaded_attn_backend = resolved_attn_backend

        if self.prewarm_var.get():
            self._prewarm_model()

        state = f"Model: loaded on {device} | {resolved_attn_backend} | {self.speed_preset_var.get()}"
        self.root.after(0, lambda: self.model_state_var.set(state))

    def _build_model_load_kwargs(
        self,
        attn_backend: str,
        token_kw: dict[str, str],
        torch_dtype: torch.dtype,
        device: str,
    ) -> dict[str, Any]:
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "attn_implementation": attn_backend,
            **token_kw,
        }

        if device == "cuda":
            total_gb = 16
            if psutil is not None:
                try:
                    total_gb = max(8, int(psutil.virtual_memory().total / (1024**3)))
                except Exception:
                    total_gb = 16

            cpu_budget = max(8, total_gb - 4)
            load_kwargs.update(
                {
                    "device_map": "auto",
                    "max_memory": {0: "3.4GiB", "cpu": f"{cpu_budget}GiB"},
                    "torch_dtype": torch_dtype,
                }
            )
        else:
            load_kwargs.update({"torch_dtype": torch.float32})

        return load_kwargs

    def _resolve_attention_backend(self, requested: str, log: bool = True) -> str:
        requested = requested or DEFAULT_ATTENTION_BACKEND
        if requested == "flash_attention_2":
            if not torch.cuda.is_available():
                if log:
                    self._log("FlashAttention requires CUDA. Falling back to SDPA.")
                return "sdpa"
            if importlib.util.find_spec("flash_attn") is None:
                if log:
                    self._log("flash_attn is not importable. Falling back to SDPA.")
                return "sdpa"
            return "flash_attention_2"
        if requested in {"sdpa", "eager"}:
            return requested
        if log:
            self._log(f"Unknown attention backend '{requested}'. Falling back to SDPA.")
        return "sdpa"

    def _prewarm_model(self):
        if self.processor is None or self.model is None:
            return

        start = time.perf_counter()
        self._log("Prewarming model...")
        payload = {
            "system_prompt": "You are a concise visual captioning engine.",
            "user_prompt": "Describe this image in one short phrase.",
            "enable_thinking": False,
            "max_new_tokens": 8,
            "max_image_side": 64,
            "do_sample": False,
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.05,
        }
        image = Image.new("RGB", (64, 64), "#101a2d")
        try:
            self._caption_image_object(image, payload, log_details=False)
            elapsed = time.perf_counter() - start
            self._log(f"Prewarm completed in {elapsed:.2f}s.")
        except Exception as exc:
            self._log(f"Prewarm skipped after error: {type(exc).__name__}: {exc}")

    def unload_model(self, silent: bool = False):
        try:
            if self.model is not None:
                del self.model
            if self.processor is not None:
                del self.processor
        except Exception:
            pass

        self.model = None
        self.processor = None
        self.loaded_model_name = None
        self.loaded_device = None
        self.loaded_dtype = None
        self.loaded_attn_backend = None

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

        self.model_state_var.set("Model: not loaded")
        if not silent:
            self._set_status("Model unloaded.", self.muted)
            self._log("Model unloaded and CUDA cache cleared.")

    # ---------- Generation Control ----------

    def request_stop(self):
        if not self.worker or not self.worker.is_alive():
            self._set_status("No active job to stop.", self.warn)
            return
        self.stop_requested.set()
        self._set_status("Stop requested. Waiting for the current step to end...", self.warn)
        self._log("Stop requested by user.")

    def generate_async(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "A job is already running.")
            return

        payload = self._collect_payload()
        error = self._validate_payload(payload)
        if error:
            messagebox.showerror("Input error", error)
            return

        self.stop_requested.clear()
        self._clear_logs()
        self._append_output("\n\n---\nStarting...\n")
        self.generate_button.config(state="disabled")
        self.stop_button.config(state="normal")

        self.worker = threading.Thread(target=self._worker_generate, args=(payload,), daemon=True)
        self.worker.start()

    def _collect_payload(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name_var.get().strip() or DEFAULT_MODEL_NAME,
            "hf_token": self.hf_token_var.get().strip(),
            "image_path": self.image_path_var.get().strip(),
            "save_path": self.save_path_var.get().strip(),
            "save_format": self.save_format_var.get().strip(),

            "batch_mode": bool(self.batch_mode_var.get()),
            "batch_folder": self.batch_folder_var.get().strip(),
            "batch_save_folder": self.batch_save_folder_var.get().strip(),
            "batch_prefix": self.batch_prefix_var.get().strip(),
            "skip_existing": bool(self.skip_existing_var.get()),

            "system_prompt": self.system_prompt_text.get("1.0", tk.END).strip() or DEFAULT_SYSTEM_PROMPT,
            "user_prompt": self.user_prompt_text.get("1.0", tk.END).strip() or DEFAULT_USER_PROMPT,

            "enable_thinking": bool(self.enable_thinking_var.get()),
            "speed_preset": self.speed_preset_var.get().strip() or DEFAULT_SPEED_PRESET,
            "prewarm": bool(self.prewarm_var.get()),
            "max_new_tokens": int(self.max_new_tokens_var.get()),
            "max_image_side": int(self.max_image_side_var.get()),

            "do_sample": bool(self.do_sample_var.get()),
            "temperature": float(self.temperature_var.get()),
            "top_p": float(self.top_p_var.get()),
            "top_k": int(self.top_k_var.get()),
            "repetition_penalty": float(self.repetition_penalty_var.get()),
            "attn_backend": self.attn_backend_var.get().strip() or "sdpa",
        }

    def _validate_payload(self, payload: dict[str, Any]) -> str:
        if payload["batch_mode"]:
            if not payload["batch_folder"]:
                return "Choose a batch image folder."
            if not Path(payload["batch_folder"]).is_dir():
                return "The batch image folder does not exist."
            if not payload["batch_save_folder"]:
                return "Choose a batch save folder."
        else:
            if not payload["image_path"]:
                return "Choose an image file."
            if not Path(payload["image_path"]).exists():
                return "The selected image file does not exist."
        return ""

    def _worker_generate(self, payload: dict[str, Any]):
        try:
            self._log("--- generation started ---")
            self._log(f"Payload model: {payload['model_name']}")
            self._log(f"Speed preset: {payload['speed_preset']}")
            self._log(f"Thinking: {payload['enable_thinking']}")
            self._log(f"Max tokens: {payload['max_new_tokens']}")
            self._log(f"Max image side: {payload['max_image_side']}")
            self._log(f"Requested attention backend: {payload['attn_backend']}")
            self._log(f"Batch mode: {payload['batch_mode']}")

            self.root.after(0, self._start_timer)
            self.root.after(0, lambda: self._set_status("Preparing model and inputs...", self.accent))

            self._load_model()

            if payload["batch_mode"]:
                result = self._run_batch(payload)
                if self.stop_requested.is_set():
                    self.root.after(0, lambda: self._finish_job(result, "Batch stopped.", self.warn))
                else:
                    self.root.after(0, lambda: self._finish_job(result, "Batch completed successfully.", self.good))
            else:
                self.root.after(0, lambda: self.current_item_var.set(f"Current: {Path(payload['image_path']).name}"))
                self.root.after(0, lambda p=payload["image_path"]: self._update_preview(p))
                caption = self._caption_one_image(Path(payload["image_path"]), payload)
                saved_info = self._save_single_output(payload, caption)
                final_text = f"Caption:\n{caption}"
                if saved_info:
                    final_text += f"\n\n{saved_info}"
                if self.stop_requested.is_set():
                    self.root.after(0, lambda: self._finish_job(final_text, "Generation stopped.", self.warn))
                else:
                    self.root.after(0, lambda: self._finish_job(final_text, "Caption generated successfully.", self.good))

        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            self._log(error_text)
            self.root.after(0, lambda: self._finish_error(error_text))
        finally:
            self.worker = None

    def _run_batch(self, payload: dict[str, Any]) -> str:
        folder = Path(payload["batch_folder"])
        save_folder = Path(payload["batch_save_folder"])
        save_folder.mkdir(parents=True, exist_ok=True)

        files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
        if not files:
            raise ValueError("No supported image files found in the selected batch folder.")

        lines = [
            f"Batch mode: {len(files)} image(s)",
            f"Source folder: {folder}",
            f"Save folder: {save_folder}",
            "",
        ]

        processed = 0
        skipped = 0
        started = time.perf_counter()

        for idx, image_path in enumerate(files, start=1):
            if self.stop_requested.is_set():
                break

            save_path = self._batch_output_path(payload, image_path, save_folder)
            if payload["skip_existing"] and save_path.exists():
                skipped += 1
                self.root.after(0, lambda n=image_path.name, i=idx, total=len(files): self.batch_progress_var.set(f"{i}/{total} Skipped: {n}"))
                lines.append(f"[{idx}/{len(files)}] Skipped existing: {image_path.name}")
                continue

            self.root.after(0, lambda p=str(image_path): self._update_preview(p))
            self.root.after(0, lambda n=image_path.name: self.current_item_var.set(f"Current: {n}"))
            self.root.after(0, lambda n=image_path.name, i=idx, total=len(files): self.batch_progress_var.set(f"{i}/{total} Processing: {n}"))
            self.root.after(0, lambda n=image_path.name, i=idx, total=len(files): self._set_status(f"Processing {i}/{total}: {n}", self.accent))

            img_start = time.perf_counter()
            caption = self._caption_one_image(image_path, payload)
            self._save_batch_output(payload, image_path, caption, save_folder)
            elapsed = time.perf_counter() - img_start

            processed += 1
            lines.append(f"[{idx}/{len(files)}] {image_path.name} ({elapsed:.2f}s)")
            lines.append(caption)
            lines.append("")

        total_elapsed = time.perf_counter() - started
        lines.append(f"Processed: {processed}")
        lines.append(f"Skipped: {skipped}")
        lines.append(f"Total time: {format_seconds(total_elapsed)}")
        return "\n".join(lines).strip()

    def _caption_one_image(self, image_path: Path, payload: dict[str, Any]) -> str:
        if self.processor is None or self.model is None:
            raise RuntimeError("Model is not loaded.")

        image = Image.open(image_path).convert("RGB")
        image = resize_image_for_vram(image, payload["max_image_side"])
        self._log(f"Opened image: {image_path}")
        self._log(f"Resized image to: {image.size}")
        return self._caption_image_object(image, payload)

    def _caption_image_object(self, image: Image.Image, payload: dict[str, Any], log_details: bool = True) -> str:
        messages = [
            {
                "role": "system",
                "content": payload["system_prompt"],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": payload["user_prompt"]},
                ],
            },
        ]

        chat_text = self._build_chat_text(messages, payload["enable_thinking"])
        if log_details:
            self._log("Chat template built.")

        if process_vision_info is not None:
            image_inputs, video_inputs = process_vision_info(messages)
            processor_kwargs = {
                "text": [chat_text],
                "images": image_inputs,
                "padding": True,
                "return_tensors": "pt",
            }
            if video_inputs is not None:
                processor_kwargs["videos"] = video_inputs
            inputs = self.processor(**processor_kwargs)
        else:
            if log_details:
                self._log("qwen_vl_utils not found. Falling back to direct processor image input.")
            inputs = self.processor(
                text=[chat_text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )

        target_device = self._pick_input_device()
        inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max(1, int(payload["max_new_tokens"])),
            "do_sample": bool(payload["do_sample"]),
            "use_cache": True,
            "repetition_penalty": float(payload["repetition_penalty"]),
            "stopping_criteria": StoppingCriteriaList([StopGenerationCriteria(self.stop_requested)]),
        }

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            eos_token_id = tokenizer.eos_token_id
            if pad_token_id is not None:
                gen_kwargs["pad_token_id"] = pad_token_id
            if eos_token_id is not None:
                gen_kwargs["eos_token_id"] = eos_token_id

        if payload["do_sample"]:
            gen_kwargs["temperature"] = float(payload["temperature"])
            gen_kwargs["top_p"] = float(payload["top_p"])
            gen_kwargs["top_k"] = int(payload["top_k"])

        if log_details:
            self._log(f"Generation kwargs: {gen_kwargs}")

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        new_ids = output_ids[:, prompt_len:]

        if tokenizer is None:
            raise RuntimeError("Processor tokenizer not available.")

        caption = tokenizer.batch_decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        caption = strip_thinking_blocks(caption)
        caption = re.sub(r"\s+", " ", caption).strip()

        if not caption:
            caption = "(No caption was generated.)"

        if log_details:
            self._log("Caption generated.")
        return caption

    def _build_chat_text(self, messages: list[dict[str, Any]], enable_thinking: bool) -> str:
        try:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            self._log("Processor apply_chat_template does not accept enable_thinking. Retrying without it.")
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _pick_input_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    # ---------- Save helpers ----------

    def _save_single_output(self, payload: dict[str, Any], caption: str) -> str:
        save_path = payload["save_path"]
        if not save_path:
            return ""

        path = Path(save_path)
        suffix = path.suffix.lower()
        if suffix not in {".txt", ".json"}:
            suffix = ".json" if payload["save_format"] == "json" else ".txt"
            path = path.with_suffix(suffix)

        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix.lower() == ".json":
            data = {
                "model_name": payload["model_name"],
                "image_path": payload["image_path"],
                "system_prompt": payload["system_prompt"],
                "user_prompt": payload["user_prompt"],
                "enable_thinking": payload["enable_thinking"],
                "speed_preset": payload.get("speed_preset", DEFAULT_SPEED_PRESET),
                "caption": caption,
                "generation": {
                    "max_new_tokens": payload["max_new_tokens"],
                    "max_image_side": payload["max_image_side"],
                    "do_sample": payload["do_sample"],
                    "temperature": payload["temperature"],
                    "top_p": payload["top_p"],
                    "top_k": payload["top_k"],
                    "repetition_penalty": payload["repetition_penalty"],
                    "requested_attn_backend": payload["attn_backend"],
                    "resolved_attn_backend": self.loaded_attn_backend,
                    "prewarm": payload.get("prewarm", False),
                },
            }
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            path.write_text(caption + "\n", encoding="utf-8")

        return f"Saved to: {path}"

    def _batch_output_path(self, payload: dict[str, Any], image_path: Path, save_folder: Path) -> Path:
        prefix = sanitize_filename(payload["batch_prefix"])
        base = image_path.stem if not prefix else f"{prefix}_{image_path.stem}"
        ext = ".json" if payload["save_format"] == "json" else ".txt"
        return save_folder / f"{base}{ext}"

    def _save_batch_output(self, payload: dict[str, Any], image_path: Path, caption: str, save_folder: Path):
        path = self._batch_output_path(payload, image_path, save_folder)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix.lower() == ".json":
            data = {
                "model_name": payload["model_name"],
                "image_path": str(image_path),
                "system_prompt": payload["system_prompt"],
                "user_prompt": payload["user_prompt"],
                "enable_thinking": payload["enable_thinking"],
                "speed_preset": payload.get("speed_preset", DEFAULT_SPEED_PRESET),
                "caption": caption,
                "generation": {
                    "max_new_tokens": payload["max_new_tokens"],
                    "max_image_side": payload["max_image_side"],
                    "do_sample": payload["do_sample"],
                    "temperature": payload["temperature"],
                    "top_p": payload["top_p"],
                    "top_k": payload["top_k"],
                    "repetition_penalty": payload["repetition_penalty"],
                    "requested_attn_backend": payload["attn_backend"],
                    "resolved_attn_backend": self.loaded_attn_backend,
                    "prewarm": payload.get("prewarm", False),
                },
            }
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            path.write_text(caption + "\n", encoding="utf-8")

    # ---------- Finish / Error ----------

    def _finish_job(self, text: str, status: str, color: str):
        self._stop_timer()
        self._set_output(text)
        self._set_status(status, color)
        self.generate_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.batch_progress_var.set("")
        self.current_item_var.set("Current: done")

    def _finish_error(self, error_text: str):
        self._stop_timer()
        self._set_output(f"Error:\n{error_text}")
        self._set_status("Generation failed. See logs and output for details.", self.bad)
        self.generate_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.batch_progress_var.set("")
        self.current_item_var.set("Current: error")
        messagebox.showerror("Generation failed", error_text.splitlines()[0])

    # ---------- Shutdown ----------

    def on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("Exit", "A job is still running. Stop it and exit?"):
                return
            self.stop_requested.set()
        self.unload_model(silent=True)
        self.root.destroy()


def main():
    root_cls = TkinterDnD.Tk if TkinterDnD is not None else tk.Tk
    root = root_cls()
    app = QwenCaptionStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
