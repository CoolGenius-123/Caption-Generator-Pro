"""
Tkinter image caption generator for LLaVA-style models.

Features:
- model name input
- output length slider
- extra generation controls
- image path picker
- optional save path for .txt or .json
- editable prompt box
- responsive UI with background generation
"""

from __future__ import annotations

import json
import hashlib
import re
import threading
import time
import subprocess
import traceback
from pathlib import Path
from typing import Any, cast
from tkinter import BOTH, END, LEFT, RIGHT, TOP, X, Y, filedialog, messagebox, scrolledtext, ttk
import tkinter as tk

try:
    import psutil
except Exception:
    psutil = None

import torch
from PIL import Image, ImageTk
from transformers import AutoProcessor, LlavaForConditionalGeneration, StoppingCriteria, StoppingCriteriaList


DEFAULT_MODEL_NAME = "fancyfeast/llama-joycaption-beta-one-hf-llava"
DEFAULT_PROMPT = (
    "Write a detailed descriptive caption for this image in a clear, natural, "
    "and formal tone. Mention the main subject, colors, setting, mood, and "
    "notable details."
)
DEFAULT_IMAGE_HINT = "Choose an image file..."
DEFAULT_SAVE_HINT = "Optional: choose where to save the caption..."
DEFAULT_CAPTION_PREFIX_HINT = "Optional: add a prefix or ending phrase..."


class PlaceholderEntry(tk.Entry):
    def __init__(self, master, placeholder: str, placeholder_fg: str, text_fg: str, **kwargs):
        super().__init__(master, **kwargs)
        self.placeholder = placeholder
        self.placeholder_fg = placeholder_fg
        self.text_fg = text_fg
        self._has_placeholder = False
        self.bind("<FocusIn>", self._clear_placeholder)
        self.bind("<FocusOut>", self._restore_placeholder)
        self._restore_placeholder()

    def _clear_placeholder(self, _event=None):
        if self._has_placeholder:
            self.delete(0, END)
            self.config(fg=self.text_fg)
            self._has_placeholder = False

    def _restore_placeholder(self, _event=None):
        if not self.get().strip():
            self.delete(0, END)
            self.insert(0, self.placeholder)
            self.config(fg=self.placeholder_fg)
            self._has_placeholder = True

    def get_value(self) -> str:
        if self._has_placeholder:
            return ""
        return self.get().strip()
class StopGenerationCriteria(StoppingCriteria):
    def __init__(self, stop_event: threading.Event):
        super().__init__()
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs) -> torch.BoolTensor:
        return cast(torch.BoolTensor, torch.tensor([self.stop_event.is_set()], device=input_ids.device, dtype=torch.bool))


class CaptionGeneratorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Caption Generator Pro")
        self.root.geometry("1140x820")
        self.root.minsize(1020, 760)
        self.root.configure(bg="#0f172a")

        self.model_cache: dict[tuple[str, str], tuple[Any, Any, str]] = {}
        self.current_worker: threading.Thread | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.brand_icon_photo: ImageTk.PhotoImage | None = None
        self.preview_image_id: int | None = None
        self.debug_visible = tk.BooleanVar(value=False)
        self.batch_mode_var = tk.BooleanVar(value=False)
        self.stop_requested = threading.Event()
        self.active_scroll_canvas: tk.Canvas | None = None

        self.caption_timer_var = tk.StringVar(value="⏱ Idle | Last Image Took: --:--:--")
        self.ram_var = tk.StringVar(value="RAM: --")
        self.cpu_var = tk.StringVar(value="CPU: --")
        self.gpu_var = tk.StringVar(value="GPU: --")
        self.cpu_temp_var = tk.StringVar(value="CPU TEMP: --")
        self.gpu_temp_var = tk.StringVar(value="GPU TEMP: --")
        self.caption_timer_running = False
        self.caption_timer_start = 0.0
        self.caption_timer_job: str | None = None
        self.last_caption_elapsed: str = "--:--:--"
        self.batch_progress_var = tk.StringVar(value="")
        self.current_image_var = tk.StringVar(value="Current image: none selected")

        self._apply_window_icon()
        self._build_styles()
        self._build_layout()
        self._update_system_info()
        self._set_status("Ready. Pick an image and press Generate.")

    def _icon_path(self) -> Path:
        return Path(__file__).resolve().parent.parent / "assets" / "logo.ico"

    def _apply_window_icon(self):
        icon_path = self._icon_path()
        if icon_path.exists():
            try:
                self.root.iconbitmap(default=str(icon_path))
            except Exception:
                pass

    def _build_styles(self):
        self.bg = "#0f172a"
        self.panel = "#162033"
        self.panel_alt = "#1c2840"
        self.accent = "#38bdf8"
        self.accent2 = "#f59e0b"
        self.text = "#e5eefb"
        self.muted = "#93a4bf"
        self.border = "#2d3a57"
        self.good = "#22c55e"
        self.error = "#ef4444"

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=self.bg)
        style.configure("Card.TFrame", background=self.panel)
        style.configure("CardAlt.TFrame", background=self.panel_alt)
        style.configure("TLabel", background=self.bg, foreground=self.text, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=self.bg, foreground=self.text, font=("Segoe UI", 22, "bold"))
        style.configure("SubTitle.TLabel", background=self.bg, foreground=self.muted, font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=self.panel, foreground=self.text, font=("Segoe UI", 11, "bold"))
        style.configure("TCheckbutton", background=self.panel, foreground=self.text, font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", self.panel)])
        style.configure(
            "Accent.TButton",
            background=self.accent,
            foreground="#08111f",
            font=("Segoe UI", 10, "bold"),
            padding=(14, 10),
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#7dd3fc"), ("disabled", "#35516b")],
            foreground=[("disabled", "#9bb0c5")],
        )
        style.configure(
            "Soft.TButton",
            background="#24324d",
            foreground=self.text,
            font=("Segoe UI", 10),
            padding=(12, 8),
        )
        style.map("Soft.TButton", background=[("active", "#334160")])
        style.configure(
            "Danger.TButton",
            background="#ef4444",
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padding=(12, 8),
        )
        style.map("Danger.TButton", background=[("active", "#f87171"), ("disabled", "#7f1d1d")])
        style.configure(
            "TCombobox",
            fieldbackground="#12203a",
            background="#12203a",
            foreground=self.text,
            arrowsize=14,
            padding=6,
        )

    def _build_layout(self):
        header = ttk.Frame(self.root, style="TFrame")
        header.pack(fill=X, padx=20, pady=(18, 8))

        title_row = tk.Frame(header, bg=self.bg)
        title_row.pack(fill=X)

        tk.Label(
            title_row,
            text="Caption Generator Pro",
            bg=self.bg,
            fg=self.text,
            font=("Segoe UI", 22, "bold"),
        ).pack(anchor="center")
        tk.Label(
            title_row,
            text="Fast local image captioning with a clean GUI for models, prompts, and save options.",
            bg=self.bg,
            fg=self.muted,
            font=("Segoe UI", 10),
        ).pack(anchor="center", pady=(4, 0))

        self._build_top_nav()

        body = ttk.Frame(self.root, style="TFrame")
        body.pack(fill=BOTH, expand=True, padx=20, pady=12)

        left = tk.Frame(body, bg=self.bg)
        left.pack(side=LEFT, fill=Y, expand=False)

        left_canvas = tk.Canvas(left, bg=self.bg, highlightthickness=0, bd=0, width=390)
        left_scrollbar = ttk.Scrollbar(left, orient="vertical", command=left_canvas.yview)
        left_scrollable = tk.Frame(left_canvas, bg=self.bg)

        left_scrollable.bind(
            "<Configure>",
            lambda _event: left_canvas.configure(scrollregion=left_canvas.bbox("all")),
        )
        left_canvas_window = left_canvas.create_window((0, 0), window=left_scrollable, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_canvas.pack(side=LEFT, fill=Y, expand=False)
        left_scrollbar.pack(side=RIGHT, fill=Y)

        def _sync_canvas_width(event):
            left_canvas.itemconfigure(left_canvas_window, width=event.width)

        left_canvas.bind("<Configure>", _sync_canvas_width)
        left_canvas.bind("<Enter>", lambda _event, canvas=left_canvas: self._set_active_scroll_canvas(canvas))

        right = tk.Frame(body, bg=self.bg)
        right.pack(side=RIGHT, fill=BOTH, expand=True, padx=(16, 0))

        right_canvas = tk.Canvas(right, bg=self.bg, highlightthickness=0, bd=0)
        right_scrollbar = ttk.Scrollbar(right, orient="vertical", command=right_canvas.yview)
        right_scrollable = tk.Frame(right_canvas, bg=self.bg)

        right_scrollable.bind(
            "<Configure>",
            lambda _event: right_canvas.configure(scrollregion=right_canvas.bbox("all")),
        )
        right_canvas_window = right_canvas.create_window((0, 0), window=right_scrollable, anchor="nw")
        right_canvas.configure(yscrollcommand=right_scrollbar.set)

        right_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        right_scrollbar.pack(side=RIGHT, fill=Y)

        def _sync_right_canvas_width(event):
            right_canvas.itemconfigure(right_canvas_window, width=event.width)

        right_canvas.bind("<Configure>", _sync_right_canvas_width)
        right_canvas.bind("<Enter>", lambda _event, canvas=right_canvas: self._set_active_scroll_canvas(canvas))

        self.root.bind_all("<MouseWheel>", self._on_mousewheel_global, add="+")

        self._build_settings_panel(left_scrollable)
        self._build_output_panel(right_scrollable)
        self._build_footer()

    def _build_top_nav(self):
        nav = tk.Frame(self.root, bg="#0b1220", highlightbackground=self.border, highlightthickness=1)
        nav.pack(fill=X, padx=20, pady=(0, 8))

        left = tk.Frame(nav, bg="#0b1220")
        left.pack(side=LEFT, padx=14, pady=10)
        tk.Label(left, text="📡 Live Monitor", bg="#0b1220", fg=self.text, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(left, text="System status updates in real time while you caption.", bg="#0b1220", fg=self.muted, font=("Segoe UI", 9)).pack(anchor="w")

        right = tk.Frame(nav, bg="#0b1220")
        right.pack(side=RIGHT, padx=14, pady=10)

        info_row = tk.Frame(right, bg="#0b1220")
        info_row.pack(anchor="e")

        self._make_info_chip(info_row, "RAM", self.ram_var)
        self._make_info_chip(info_row, "CPU", self.cpu_var)
        self._make_info_chip(info_row, "GPU", self.gpu_var)
        self._make_info_chip(info_row, "CPU TEMP", self.cpu_temp_var)
        self._make_info_chip(info_row, "GPU TEMP", self.gpu_temp_var)

        clock_box = tk.Frame(right, bg="#132033", highlightbackground=self.border, highlightthickness=1)
        clock_box.pack(anchor="e", pady=(8, 0), fill=X)
        timer_row = tk.Frame(clock_box, bg="#132033")
        timer_row.pack(fill=X)
        tk.Label(timer_row, text="⏱", bg="#132033", fg=self.accent, font=("Segoe UI", 12)).pack(side=LEFT, padx=(10, 6), pady=(6, 2))
        tk.Label(timer_row, textvariable=self.caption_timer_var, bg="#132033", fg=self.text, font=("Segoe UI", 12, "bold")).pack(side=LEFT, padx=(0, 10), pady=(6, 2))
        tk.Label(clock_box, textvariable=self.batch_progress_var, bg="#132033", fg=self.muted, font=("Segoe UI", 9)).pack(anchor="w", padx=10, pady=(0, 6))
        tk.Label(clock_box, textvariable=self.current_image_var, bg="#132033", fg=self.muted, font=("Segoe UI", 9)).pack(anchor="w", padx=10, pady=(0, 6))

    def _make_info_chip(self, parent: tk.Widget, title: str, value_var: tk.StringVar):
        chip = tk.Frame(parent, bg="#132033", highlightbackground=self.border, highlightthickness=1)
        chip.pack(side=LEFT, padx=(0, 8))
        tk.Label(chip, text=title, bg="#132033", fg=self.muted, font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=8, pady=(5, 0))
        tk.Label(chip, textvariable=value_var, bg="#132033", fg=self.text, font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=8, pady=(0, 5))

    def _set_active_scroll_canvas(self, canvas: tk.Canvas):
        self.active_scroll_canvas = canvas

    def _on_mousewheel_global(self, event):
        canvas = self.active_scroll_canvas
        if canvas is None or event.widget.winfo_toplevel() is not self.root:
            return
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _update_system_info(self):
        self.ram_var.set(self._get_ram_text())
        self.cpu_var.set(self._get_cpu_text())
        self.gpu_var.set(self._get_gpu_text())
        self.cpu_temp_var.set(self._get_cpu_temp_text())
        self.gpu_temp_var.set(self._get_gpu_temp_text())
        self.root.after(1000, self._update_system_info)

    def _format_elapsed(self, seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"⏱ {hours:02d}:{minutes:02d}:{secs:02d}"

    def _start_caption_timer(self):
        self.caption_timer_running = True
        self.caption_timer_start = time.perf_counter()
        self.caption_timer_var.set(f"⏱ Running | Last Image Took: {self.last_caption_elapsed}")
        self._update_caption_timer()

    def _update_caption_timer(self):
        if not self.caption_timer_running:
            return
        elapsed = time.perf_counter() - self.caption_timer_start
        self.caption_timer_var.set(f"⏱ Running {self._format_elapsed(elapsed)} | Last Image Took: {self.last_caption_elapsed}")
        self.caption_timer_job = self.root.after(250, self._update_caption_timer)

    def _stop_caption_timer(self, keep_elapsed: bool = True):
        if self.caption_timer_job is not None:
            try:
                self.root.after_cancel(self.caption_timer_job)
            except Exception:
                pass
            self.caption_timer_job = None
        elapsed = time.perf_counter() - self.caption_timer_start if self.caption_timer_running else 0.0
        self.caption_timer_running = False
        if keep_elapsed:
            elapsed_text = self._format_elapsed(elapsed)
            self.last_caption_elapsed = elapsed_text.replace("⏱ ", "")
        self.caption_timer_var.set(f"⏱ Idle | Last Image Took: {self.last_caption_elapsed}")

    def _reset_caption_timer(self):
        self._stop_caption_timer(keep_elapsed=False)

    def _get_ram_text(self) -> str:
        if psutil is None:
            return "RAM: N/A"
        vm = psutil.virtual_memory()
        return f"RAM: {vm.percent:.0f}%"

    def _get_cpu_text(self) -> str:
        if psutil is None:
            return "CPU: N/A"
        return f"CPU: {psutil.cpu_percent(interval=None):.0f}%"

    def _get_cpu_temp_text(self) -> str:
        temp = self._read_cpu_temp()
        return f"CPU TEMP: {temp}" if temp else "CPU TEMP: N/A"

    def _read_cpu_temp(self) -> str | None:
        sensors_temperatures = getattr(psutil, "sensors_temperatures", None) if psutil is not None else None
        if callable(sensors_temperatures):
            try:
                temps = sensors_temperatures(fahrenheit=False)
                temp_groups = cast(dict[Any, Any], temps).values()
                for entries in temp_groups:
                    for entry in entries:
                        if entry.current is not None:
                            return f"{entry.current:.0f}°C"
            except Exception:
                pass

        try:
            result = subprocess.run(
                ["wmic", "/namespace:\\root\\wmi", "PATH", "MSAcpi_ThermalZoneTemperature", "get", "CurrentTemperature"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            values = []
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.isdigit():
                    values.append(int(line))
            if values:
                celsius = (values[0] / 10.0) - 273.15
                return f"{celsius:.0f}°C"
        except Exception:
            pass
        return None

    def _get_gpu_text(self) -> str:
        if not torch.cuda.is_available():
            return "GPU: N/A"
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            value = result.stdout.strip().splitlines()[0].strip()
            return f"GPU: {value}%"
        except Exception:
            try:
                return f"GPU: {torch.cuda.get_device_name(0)}"
            except Exception:
                return "GPU: CUDA"

    def _get_gpu_temp_text(self) -> str:
        if not torch.cuda.is_available():
            return "GPU TEMP: N/A"
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            value = result.stdout.strip().splitlines()[0].strip()
            return f"GPU TEMP: {value}°C"
        except Exception:
            return "GPU TEMP: N/A"

    def _build_settings_panel(self, parent: tk.Frame):
        panel = tk.Frame(parent, bg=self.panel, bd=0, highlightbackground=self.border, highlightthickness=1)
        panel.pack(fill=Y, expand=False)

        top = tk.Frame(panel, bg=self.panel)
        top.pack(fill=X, padx=16, pady=(16, 10))
        tk.Label(top, text="⚙️ Settings", bg=self.panel, fg=self.text, font=("Segoe UI", 14, "bold")).pack(anchor="w")
        tk.Label(
            top,
            text="Use a LLaVA-compatible vision model for best results.",
            bg=self.panel,
            fg=self.muted,
            font=("Segoe UI", 9),
            wraplength=340,
            justify="left",
        ).pack(anchor="w", pady=(3, 0))

        form = tk.Frame(panel, bg=self.panel)
        form.pack(fill=BOTH, expand=True, padx=16, pady=(0, 14))

        self.model_name_var = tk.StringVar(value=DEFAULT_MODEL_NAME)
        self.hf_token_var = tk.StringVar(value="")
        self.image_path_var = tk.StringVar()
        self.save_path_var = tk.StringVar()
        self.batch_image_folder_var = tk.StringVar()
        self.batch_save_folder_var = tk.StringVar()
        self.batch_prefix_var = tk.StringVar()
        self.caption_prefix_position_var = tk.StringVar(value="front")
        self.save_format_var = tk.StringVar(value="txt")
        self.output_length_var = tk.IntVar(value=128)
        self.temperature_var = tk.DoubleVar(value=0.6)
        self.top_p_var = tk.DoubleVar(value=0.9)
        self.top_k_var = tk.IntVar(value=40)
        self.repetition_var = tk.DoubleVar(value=1.05)
        self.do_sample_var = tk.BooleanVar(value=False)

        self._section_label(form, "🧠 Model name")
        self.model_entry = tk.Entry(
            form,
            textvariable=self.model_name_var,
            bg="#12203a",
            fg=self.text,
            insertbackground=self.text,
            relief="flat",
            font=("Segoe UI", 10),
        )
        self.model_entry.pack(fill=X, pady=(6, 12), ipady=7)

        self._section_label(form, "🔐 Hugging Face token (optional)")
        self.hf_token_entry = PlaceholderEntry(
            form,
            placeholder="hf_...",
            placeholder_fg=self.muted,
            text_fg=self.text,
            bg="#12203a",
            fg=self.text,
            insertbackground=self.text,
            relief="flat",
            font=("Segoe UI", 10),
        )
        self.hf_token_entry.pack(fill=X, pady=(6, 12), ipady=7)

        self._build_preview_card(form)

        self._section_label(form, "🖼️ Image path")
        image_row = tk.Frame(form, bg=self.panel)
        image_row.pack(fill=X, pady=(6, 12))
        self.image_entry = PlaceholderEntry(
            image_row,
            placeholder=DEFAULT_IMAGE_HINT,
            placeholder_fg=self.muted,
            text_fg=self.text,
            bg="#12203a",
            fg=self.text,
            insertbackground=self.text,
            relief="flat",
            font=("Segoe UI", 10),
        )
        self.image_entry.pack(side=LEFT, fill=X, expand=True, ipady=7)
        self.image_entry.bind("<FocusOut>", lambda _event: self._update_preview_from_entry())
        tk.Button(
            image_row,
            text="📂",
            command=self._choose_image,
            bg="#263754",
            fg=self.text,
            activebackground="#334866",
            activeforeground=self.text,
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            width=4,
        ).pack(side=RIGHT, padx=(8, 0), ipady=4)

        self.batch_toggle = tk.Checkbutton(
            form,
            text="🗂️ Enable multi-image folder mode",
            variable=self.batch_mode_var,
            command=self._toggle_batch_mode,
            bg=self.panel,
            fg=self.text,
            activebackground=self.panel,
            activeforeground=self.text,
            selectcolor=self.panel_alt,
            font=("Segoe UI", 10),
            relief="flat",
        )
        self.batch_toggle.pack(anchor="w", pady=(2, 8))

        self.batch_frame = tk.Frame(form, bg=self.panel)
        self._build_batch_controls(self.batch_frame)
        self.batch_frame.pack_forget()

        self._section_label(form, "💾 Save path (optional)")
        save_row = tk.Frame(form, bg=self.panel)
        save_row.pack(fill=X, pady=(6, 8))
        self.save_entry = PlaceholderEntry(
            save_row,
            placeholder=DEFAULT_SAVE_HINT,
            placeholder_fg=self.muted,
            text_fg=self.text,
            bg="#12203a",
            fg=self.text,
            insertbackground=self.text,
            relief="flat",
            font=("Segoe UI", 10),
        )
        self.save_entry.pack(side=LEFT, fill=X, expand=True, ipady=7)
        tk.Button(
            save_row,
            text="💾",
            command=self._choose_save_path,
            bg="#263754",
            fg=self.text,
            activebackground="#334866",
            activeforeground=self.text,
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            width=4,
        ).pack(side=RIGHT, padx=(8, 0), ipady=4)

        save_format_row = tk.Frame(form, bg=self.panel)
        save_format_row.pack(fill=X, pady=(0, 10))
        tk.Label(save_format_row, text="Format", bg=self.panel, fg=self.muted, font=("Segoe UI", 9)).pack(side=LEFT)
        save_format = ttk.Combobox(
            save_format_row,
            textvariable=self.save_format_var,
            values=("txt", "json"),
            state="readonly",
            width=10,
        )
        save_format.pack(side=RIGHT)

        self._section_label(form, "🗣️ Caption prefix (optional)")
        caption_prefix_row = tk.Frame(form, bg=self.panel)
        caption_prefix_row.pack(fill=X, pady=(6, 6))
        self.caption_prefix_entry = PlaceholderEntry(
            caption_prefix_row,
            placeholder=DEFAULT_CAPTION_PREFIX_HINT,
            placeholder_fg=self.muted,
            text_fg=self.text,
            bg="#12203a",
            fg=self.text,
            insertbackground=self.text,
            relief="flat",
            font=("Segoe UI", 10),
        )
        self.caption_prefix_entry.pack(side=LEFT, fill=X, expand=True, ipady=7)

        caption_prefix_position = ttk.Combobox(
            caption_prefix_row,
            textvariable=self.caption_prefix_position_var,
            values=("front", "last"),
            state="readonly",
            width=10,
        )
        caption_prefix_position.pack(side=RIGHT, padx=(8, 0))

        self._section_label(form, "✍️ Prompt")
        prompt_hint = tk.Label(
            form,
            text="Edit this only if you want a custom prompt template.",
            bg=self.panel,
            fg=self.muted,
            font=("Segoe UI", 9),
            wraplength=340,
            justify="left",
        )
        prompt_hint.pack(anchor="w", pady=(4, 4))

        self.prompt_text = scrolledtext.ScrolledText(
            form,
            height=7,
            wrap=tk.WORD,
            bg="#12203a",
            fg=self.text,
            insertbackground=self.text,
            relief="flat",
            font=("Segoe UI", 10),
        )
        self.prompt_text.pack(fill=X, pady=(0, 12))
        self.prompt_text.insert("1.0", DEFAULT_PROMPT)

        self._section_label(form, "📏 Output length")
        length_row = tk.Frame(form, bg=self.panel)
        length_row.pack(fill=X, pady=(6, 2))
        self.length_value_label = tk.Label(length_row, text="128", bg=self.panel, fg=self.accent, font=("Segoe UI", 10, "bold"))
        self.length_value_label.pack(side=RIGHT)
        self.length_scale = tk.Scale(
            form,
            from_=0,
            to=512,
            orient="horizontal",
            variable=self.output_length_var,
            bg=self.panel,
            fg=self.text,
            troughcolor="#24324d",
            highlightthickness=0,
            activebackground=self.accent,
            relief="flat",
            command=self._sync_length_label,
        )
        self.length_scale.pack(fill=X, pady=(0, 10))

        self._section_label(form, "🎛️ Extra generation values")
        self._slider_block(form, "Temperature", self.temperature_var, 0.0, 2.0, 0.05, "0.60")
        self._slider_block(form, "Top-p", self.top_p_var, 0.0, 1.0, 0.01, "0.90")
        self._slider_block(form, "Top-k", self.top_k_var, 0, 100, 1, "40")
        self._slider_block(form, "Repetition penalty", self.repetition_var, 1.0, 2.0, 0.01, "1.05")

        self.sample_check = tk.Checkbutton(
            form,
            text="🎲 Use sampling (slower, more varied)",
            variable=self.do_sample_var,
            bg=self.panel,
            fg=self.text,
            activebackground=self.panel,
            activeforeground=self.text,
            selectcolor=self.panel_alt,
            font=("Segoe UI", 10),
            relief="flat",
        )
        self.sample_check.pack(anchor="w", pady=(8, 12))

        self.generate_button = ttk.Button(form, text="✨ Generate Caption", style="Accent.TButton", command=self.generate_caption)
        self.generate_button.pack(fill=X, pady=(8, 0), ipady=2)

        self.stop_button = ttk.Button(form, text="🛑 Stop", style="Danger.TButton", command=self.stop_captioning, state="disabled")
        self.stop_button.pack(fill=X, pady=(8, 0), ipady=2)

        quick_row = tk.Frame(form, bg=self.panel)
        quick_row.pack(fill=X, pady=(10, 0))
        ttk.Button(quick_row, text="🧹 Clear Output", style="Soft.TButton", command=self._clear_output).pack(side=LEFT, fill=X, expand=True)
        ttk.Button(quick_row, text="🔁 Reset Defaults", style="Soft.TButton", command=self._reset_defaults).pack(side=LEFT, fill=X, expand=True, padx=(8, 0))

    def _build_batch_controls(self, parent: tk.Frame):
        card = tk.Frame(parent, bg=self.panel_alt, bd=0, highlightbackground=self.border, highlightthickness=1)
        card.pack(fill=X, pady=(0, 12))

        header = tk.Frame(card, bg=self.panel_alt)
        header.pack(fill=X, padx=12, pady=(10, 6))
        tk.Label(header, text="📚 Multi-image mode", bg=self.panel_alt, fg=self.text, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(
            header,
            text="Choose a folder of images, then choose where the captions should be saved.",
            bg=self.panel_alt,
            fg=self.muted,
            font=("Segoe UI", 9),
            wraplength=320,
            justify="left",
        ).pack(anchor="w", pady=(2, 0))

        source_row = tk.Frame(card, bg=self.panel_alt)
        source_row.pack(fill=X, padx=12, pady=(6, 8))
        tk.Label(source_row, text="Images folder", bg=self.panel_alt, fg=self.text, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        folder_row = tk.Frame(source_row, bg=self.panel_alt)
        folder_row.pack(fill=X, pady=(5, 0))
        self.batch_image_folder_entry = PlaceholderEntry(
            folder_row,
            placeholder="Choose a folder with images...",
            placeholder_fg=self.muted,
            text_fg=self.text,
            bg="#12203a",
            fg=self.text,
            insertbackground=self.text,
            relief="flat",
            font=("Segoe UI", 10),
        )
        self.batch_image_folder_entry.pack(side=LEFT, fill=X, expand=True, ipady=7)
        tk.Button(
            folder_row,
            text="📁",
            command=self._choose_batch_images_folder,
            bg="#263754",
            fg=self.text,
            activebackground="#334866",
            activeforeground=self.text,
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            width=4,
        ).pack(side=RIGHT, padx=(8, 0), ipady=4)

        save_row = tk.Frame(card, bg=self.panel_alt)
        save_row.pack(fill=X, padx=12, pady=(4, 8))
        tk.Label(save_row, text="Save captions folder", bg=self.panel_alt, fg=self.text, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        save_folder_row = tk.Frame(save_row, bg=self.panel_alt)
        save_folder_row.pack(fill=X, pady=(5, 0))
        self.batch_save_folder_entry = PlaceholderEntry(
            save_folder_row,
            placeholder="Choose a folder for captions...",
            placeholder_fg=self.muted,
            text_fg=self.text,
            bg="#12203a",
            fg=self.text,
            insertbackground=self.text,
            relief="flat",
            font=("Segoe UI", 10),
        )
        self.batch_save_folder_entry.pack(side=LEFT, fill=X, expand=True, ipady=7)
        tk.Button(
            save_folder_row,
            text="📁",
            command=self._choose_batch_save_folder,
            bg="#263754",
            fg=self.text,
            activebackground="#334866",
            activeforeground=self.text,
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            width=4,
        ).pack(side=RIGHT, padx=(8, 0), ipady=4)

        prefix_row = tk.Frame(card, bg=self.panel_alt)
        prefix_row.pack(fill=X, padx=12, pady=(2, 12))
        tk.Label(prefix_row, text="Filename prefix", bg=self.panel_alt, fg=self.text, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.batch_prefix_entry = PlaceholderEntry(
            prefix_row,
            placeholder="Optional, for example: photo",
            placeholder_fg=self.muted,
            text_fg=self.text,
            bg="#12203a",
            fg=self.text,
            insertbackground=self.text,
            relief="flat",
            font=("Segoe UI", 10),
        )
        self.batch_prefix_entry.pack(fill=X, pady=(5, 0), ipady=7)

    def _toggle_batch_mode(self):
        if self.batch_mode_var.get():
            self.batch_frame.pack(fill=X, pady=(0, 4))
            self.generate_button.config(text="✨ Generate Batch Captions")
        else:
            self.batch_frame.pack_forget()
            self.generate_button.config(text="✨ Generate Caption")

    def _build_preview_card(self, parent: tk.Widget):
        card = tk.Frame(parent, bg=self.panel_alt, bd=0, highlightbackground=self.border, highlightthickness=1)
        card.pack(fill=X, pady=(0, 12))

        header = tk.Frame(card, bg=self.panel_alt)
        header.pack(fill=X, padx=12, pady=(10, 6))
        tk.Label(header, text="🖼️ Image Preview", bg=self.panel_alt, fg=self.text, font=("Segoe UI", 11, "bold")).pack(side=LEFT)
        tk.Label(header, text="thumbnail", bg=self.panel_alt, fg=self.muted, font=("Segoe UI", 9)).pack(side=RIGHT)

        self.preview_container = tk.Frame(card, bg="#0d1424")
        self.preview_container.pack(fill=X, padx=12, pady=(0, 12))
        self.preview_canvas = tk.Canvas(
            self.preview_container,
            width=340,
            height=240,
            bg="#0d1424",
            highlightthickness=1,
            highlightbackground=self.border,
            bd=0,
        )
        self.preview_canvas.pack(fill=X, padx=10, pady=10)
        self._clear_preview()

    def _build_output_panel(self, parent: tk.Frame):
        panel = tk.Frame(parent, bg=self.panel, bd=0, highlightbackground=self.border, highlightthickness=1)
        panel.pack(fill=BOTH, expand=True)

        top = tk.Frame(panel, bg=self.panel)
        top.pack(fill=X, padx=16, pady=(16, 8))
        tk.Label(top, text="📝 Caption Output", bg=self.panel, fg=self.text, font=("Segoe UI", 14, "bold")).pack(anchor="w")
        tk.Label(
            top,
            text="Your generated caption will appear here, and can also be saved to disk if you choose a save path.",
            bg=self.panel,
            fg=self.muted,
            font=("Segoe UI", 9),
            wraplength=620,
            justify="left",
        ).pack(anchor="w", pady=(3, 0))

        self.output_text = scrolledtext.ScrolledText(
            panel,
            wrap=tk.WORD,
            bg="#0d1424",
            fg=self.text,
            insertbackground=self.text,
            relief="flat",
            font=("Segoe UI", 11),
        )
        self.output_text.pack(fill=BOTH, expand=True, padx=16, pady=(0, 14))
        self.output_text.insert("1.0", "Ready. Choose an image and click Generate.")
        self.output_text.configure(state="disabled")

        self.status_label = tk.Label(
            panel,
            text="",
            bg=self.panel,
            fg=self.muted,
            anchor="w",
            font=("Segoe UI", 10),
        )
        self.status_label.pack(fill=X, padx=16, pady=(0, 10))

        debug_row = tk.Frame(panel, bg=self.panel)
        debug_row.pack(fill=X, padx=16, pady=(0, 8))
        self.debug_toggle = tk.Checkbutton(
            debug_row,
            text="🐞 Show debug logs",
            variable=self.debug_visible,
            command=self._toggle_debug_logs,
            bg=self.panel,
            fg=self.text,
            activebackground=self.panel,
            activeforeground=self.text,
            selectcolor=self.panel_alt,
            font=("Segoe UI", 10),
            relief="flat",
        )
        self.debug_toggle.pack(anchor="w")

        self.debug_panel = tk.Frame(panel, bg="#0d1424", bd=0, highlightbackground=self.border, highlightthickness=1)
        self.debug_header = tk.Label(
            self.debug_panel,
            text="🔎 Debug Logs",
            bg="#0d1424",
            fg=self.accent,
            anchor="w",
            font=("Segoe UI", 10, "bold"),
        )
        self.debug_header.pack(fill=X, padx=10, pady=(8, 4))
        self.debug_text = scrolledtext.ScrolledText(
            self.debug_panel,
            height=8,
            wrap=tk.WORD,
            bg="#08111f",
            fg="#8be9fd",
            insertbackground=self.text,
            relief="flat",
            font=("Consolas", 9),
        )
        self.debug_text.pack(fill=BOTH, expand=True, padx=10, pady=(0, 10))
        self.debug_text.configure(state="disabled")
        self._toggle_debug_logs()

    def _build_footer(self):
        footer = tk.Frame(self.root, bg=self.bg)
        footer.pack(fill=X, padx=20, pady=(0, 14))
        tk.Label(
            footer,
            text="Tip: shorter output is faster; sampling adds variety but usually slows generation a bit.",
            bg=self.bg,
            fg=self.muted,
            font=("Segoe UI", 9),
        ).pack(anchor="w")

    def _section_label(self, parent: tk.Widget, text: str):
        tk.Label(parent, text=text, bg=self.panel, fg=self.text, font=("Segoe UI", 11, "bold")).pack(anchor="w")

    def _slider_block(self, parent: tk.Widget, label: str, variable, from_: float, to: float, resolution: float, value_text: str):
        block = tk.Frame(parent, bg=self.panel)
        block.pack(fill=X, pady=(6, 0))
        row = tk.Frame(block, bg=self.panel)
        row.pack(fill=X)
        tk.Label(row, text=f"{label}", bg=self.panel, fg=self.text, font=("Segoe UI", 10)).pack(side=LEFT)
        value_label = tk.Label(row, text=value_text, bg=self.panel, fg=self.accent, font=("Segoe UI", 10, "bold"))
        value_label.pack(side=RIGHT)

        slider = tk.Scale(
            block,
            from_=from_,
            to=to,
            resolution=resolution,
            orient="horizontal",
            variable=variable,
            bg=self.panel,
            fg=self.text,
            troughcolor="#24324d",
            highlightthickness=0,
            activebackground=self.accent,
            relief="flat",
            command=lambda value, label_widget=value_label, kind=label: label_widget.config(text=self._format_slider_value(kind, value)),
        )
        slider.pack(fill=X, pady=(2, 4))

    def _format_slider_value(self, label: str, value: str) -> str:
        if label == "Top-k" or label == "Output length":
            return str(int(float(value)))
        return f"{float(value):.2f}"

    def _sync_length_label(self, value: str):
        self.length_value_label.config(text=str(int(float(value))))

    def _choose_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.image_entry._clear_placeholder()
            self.image_entry.delete(0, END)
            self.image_entry.insert(0, path)
            self.image_entry.config(fg=self.text)
            self._update_preview(path)
            self._set_current_image(path)

    def _choose_save_path(self):
        default_ext = ".json" if self.save_format_var.get() == "json" else ".txt"
        path = filedialog.asksaveasfilename(
            title="Save caption as",
            defaultextension=default_ext,
            filetypes=[("Text file", "*.txt"), ("JSON file", "*.json")],
        )
        if path:
            self.save_entry._clear_placeholder()
            self.save_entry.delete(0, END)
            self.save_entry.insert(0, path)
            self.save_entry.config(fg=self.text)
            suffix = Path(path).suffix.lower().lstrip(".")
            if suffix in {"txt", "json"}:
                self.save_format_var.set(suffix)

    def _choose_batch_images_folder(self):
        folder = filedialog.askdirectory(title="Choose a folder with images")
        if folder:
            self.batch_image_folder_entry._clear_placeholder()
            self.batch_image_folder_entry.delete(0, END)
            self.batch_image_folder_entry.insert(0, folder)
            self.batch_image_folder_entry.config(fg=self.text)

    def _choose_batch_save_folder(self):
        folder = filedialog.askdirectory(title="Choose a folder to save captions")
        if folder:
            self.batch_save_folder_entry._clear_placeholder()
            self.batch_save_folder_entry.delete(0, END)
            self.batch_save_folder_entry.insert(0, folder)
            self.batch_save_folder_entry.config(fg=self.text)

    def _reset_defaults(self):
        self.model_name_var.set(DEFAULT_MODEL_NAME)
        self.output_length_var.set(128)
        self._sync_length_label("128")
        self.temperature_var.set(0.6)
        self.top_p_var.set(0.9)
        self.top_k_var.set(40)
        self.repetition_var.set(1.05)
        self.do_sample_var.set(False)
        self.prompt_text.delete("1.0", END)
        self.prompt_text.insert("1.0", DEFAULT_PROMPT)
        self._clear_entry(self.image_entry, DEFAULT_IMAGE_HINT)
        self._clear_entry(self.hf_token_entry, "hf_...")
        self._clear_entry(self.save_entry, DEFAULT_SAVE_HINT)
        self._clear_entry(self.caption_prefix_entry, DEFAULT_CAPTION_PREFIX_HINT)
        self._clear_entry(self.batch_image_folder_entry, "Choose a folder with images...")
        self._clear_entry(self.batch_save_folder_entry, "Choose a folder for captions...")
        self._clear_entry(self.batch_prefix_entry, "Optional, for example: photo")
        self.caption_prefix_position_var.set("front")
        if self.batch_mode_var.get():
            self._toggle_batch_mode()
        self._clear_preview()
        self._set_status("Restored default values.")

    def _clear_entry(self, entry: PlaceholderEntry, placeholder: str):
        entry.delete(0, END)
        entry.insert(0, placeholder)
        entry.config(fg=self.muted)
        entry._has_placeholder = True

    def _clear_output(self):
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", END)
        self.output_text.insert("1.0", "Ready. Choose an image and click Generate.")
        self.output_text.configure(state="disabled")
        self._set_status("Output cleared.")

    def _set_batch_progress(self, text: str = ""):
        self.batch_progress_var.set(text)

    def _set_batch_progress_async(self, text: str = ""):
        self.root.after(0, lambda value=text: self._set_batch_progress(value))

    def _set_current_image(self, text: str):
        display_text = text.strip() if text.strip() else "none selected"
        self.current_image_var.set(f"Current image: {display_text}")

    def _set_current_image_async(self, text: str):
        self.root.after(0, lambda value=text: self._set_current_image(value))

    def _toggle_debug_logs(self):
        if self.debug_visible.get():
            self.debug_panel.pack(fill=X, padx=16, pady=(0, 14))
        else:
            self.debug_panel.pack_forget()

    def _debug_log(self, message: str):
        if not self.debug_visible.get():
            return

        def append():
            self.debug_text.configure(state="normal")
            self.debug_text.insert(END, message.rstrip() + "\n")
            self.debug_text.see(END)
            self.debug_text.configure(state="disabled")

        self.root.after(0, append)

    def _clear_debug_logs(self):
        def clear():
            self.debug_text.configure(state="normal")
            self.debug_text.delete("1.0", END)
            self.debug_text.configure(state="disabled")

        self.root.after(0, clear)

    def _update_preview_from_entry(self):
        path = self.image_entry.get_value()
        if path:
            self._update_preview(path)
        else:
            self._clear_preview()

    def _set_preview_async(self, image_path: str):
        self.root.after(0, lambda path=image_path: self._update_preview(path))

    def _update_preview(self, image_path: str):
        try:
            image_file = Path(image_path)
            if not image_file.exists():
                self._clear_preview("Image not found")
                return

            image = Image.open(image_file).convert("RGB")
            image.thumbnail((340, 240))
            self.preview_photo = ImageTk.PhotoImage(image)
            self.preview_canvas.delete("all")
            canvas_width = int(self.preview_canvas.cget("width"))
            canvas_height = int(self.preview_canvas.cget("height"))
            x_pos = canvas_width // 2
            y_pos = canvas_height // 2
            self.preview_canvas.create_image(x_pos, y_pos, image=self.preview_photo, anchor="center")
        except Exception as exc:
            self._clear_preview(f"Preview failed\n{exc}")

    def _clear_preview(self, text: str = "No image loaded yet"):
        self.preview_photo = None
        if hasattr(self, "preview_canvas"):
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(
                170,
                120,
                text=text,
                fill=self.muted,
                font=("Segoe UI", 10),
                justify="center",
            )

    def _set_status_async(self, text: str, color: str | None = None):
        self.root.after(0, lambda value=text, tint=color: self._set_status(value, tint))

    def _build_brand_icon(self, parent: tk.Widget):
        icon_path = self._icon_path()
        if icon_path.exists():
            try:
                icon_image = Image.open(icon_path).convert("RGBA")
                try:
                    resample_filter = Image.Resampling.LANCZOS
                except AttributeError:
                    resample_filter = cast(Any, Image).LANCZOS
                icon_image = icon_image.resize((40, 40), resample_filter)
                self.brand_icon_photo = ImageTk.PhotoImage(icon_image)
                tk.Label(parent, image=self.brand_icon_photo, bg=self.bg).pack(side=LEFT)
                return
            except Exception:
                pass

        tk.Label(parent, text="🖼️", bg=self.bg, fg=self.text, font=("Segoe UI", 22, "bold")).pack(side=LEFT)

    def _set_status(self, text: str, color: str | None = None):
        if color is None:
            color = self.muted
        self.status_label.config(text=text, fg=color)

    def _set_output(self, text: str):
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", END)
        self.output_text.insert("1.0", text)
        self.output_text.configure(state="disabled")

    def _append_output(self, text: str):
        self.output_text.configure(state="normal")
        self.output_text.insert(END, text)
        self.output_text.see(END)
        self.output_text.configure(state="disabled")

    def _read_prompt(self) -> str:
        prompt = self.prompt_text.get("1.0", END).strip()
        return prompt or DEFAULT_PROMPT

    def _collect_inputs(self):
        model_name = self.model_name_var.get().strip() or DEFAULT_MODEL_NAME
        hf_token = self.hf_token_entry.get_value()
        image_path = self.image_entry.get_value()
        save_path = self.save_entry.get_value()
        caption_prefix = self.caption_prefix_entry.get_value()
        caption_prefix_position = self.caption_prefix_position_var.get().strip().lower() or "front"
        batch_image_folder = self.batch_image_folder_entry.get_value()
        batch_save_folder = self.batch_save_folder_entry.get_value()
        batch_prefix = self.batch_prefix_entry.get_value()
        prompt = self._read_prompt()
        max_new_tokens = int(self.output_length_var.get())
        temperature = float(self.temperature_var.get())
        top_p = float(self.top_p_var.get())
        top_k = int(self.top_k_var.get())
        repetition_penalty = float(self.repetition_var.get())
        do_sample = bool(self.do_sample_var.get())
        return {
            "model_name": model_name,
            "hf_token": hf_token,
            "image_path": image_path,
            "save_path": save_path,
            "caption_prefix": caption_prefix,
            "caption_prefix_position": caption_prefix_position,
            "batch_mode": bool(self.batch_mode_var.get()),
            "batch_image_folder": batch_image_folder,
            "batch_save_folder": batch_save_folder,
            "batch_prefix": batch_prefix,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
        }

    def generate_caption(self):
        if self.current_worker and self.current_worker.is_alive():
            messagebox.showinfo("Busy", "Caption generation is already running.")
            return

        self.stop_requested.clear()
        payload = self._collect_inputs()
        if payload["batch_mode"]:
            if not payload["batch_image_folder"]:
                messagebox.showerror("Missing folder", "Please choose a folder of images first.")
                return
            image_folder = Path(payload["batch_image_folder"])
            if not image_folder.exists() or not image_folder.is_dir():
                messagebox.showerror("Folder not found", f"The selected images folder does not exist:\n{image_folder}")
                return
            if not payload["batch_save_folder"]:
                messagebox.showerror("Missing save folder", "Please choose a folder to save the captions.")
                return
        else:
            if not payload["image_path"]:
                messagebox.showerror("Missing image", "Please choose an image file first.")
                return

            image_file = Path(payload["image_path"])
            if not image_file.exists():
                messagebox.showerror("Image not found", f"The selected image does not exist:\n{image_file}")
                return

        self.generate_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self._set_status("Loading model and generating caption...", self.accent)
        self._set_batch_progress("")
        if payload["batch_mode"]:
            self._set_current_image(payload["batch_image_folder"])
        else:
            self._set_current_image(payload["image_path"])
        start_message = "Starting batch generation..." if payload["batch_mode"] else "Starting generation..."
        self._append_output(f"\n\n---\n{start_message}\n")
        self._clear_debug_logs()
        self._debug_log("--- generation started ---")
        self._debug_log(f"model: {payload['model_name']}")
        if payload["batch_mode"]:
            self._debug_log(f"batch folder: {payload['batch_image_folder']}")
            self._debug_log(f"save folder: {payload['batch_save_folder']}")
            self._debug_log(f"prefix: {payload['batch_prefix'] or '(none)'}")
        else:
            self._debug_log(f"image: {payload['image_path']}")
        self._debug_log(f"max_new_tokens: {payload['max_new_tokens']}")
        self._debug_log(f"do_sample: {payload['do_sample']}")

        self.current_worker = threading.Thread(target=self._worker_generate, args=(payload,), daemon=True)
        self.current_worker.start()

    def stop_captioning(self):
        if not self.current_worker or not self.current_worker.is_alive():
            messagebox.showinfo("Nothing running", "No captioning job is currently running.")
            return
        self.stop_requested.set()
        self._debug_log("stop requested by user")
        self._set_status("Stop requested. Finishing current step...", self.accent2)

    def _worker_generate(self, payload: dict):
        try:
            self._debug_log("loading processor and model")
            processor, model, device = self._load_model(payload["model_name"], payload["hf_token"])
            self._debug_log(f"using device: {device}")
            if payload["batch_mode"]:
                result_text, summary, stopped = self._generate_batch_captions(payload, processor, model, device)
                self._debug_log(summary)
                if stopped:
                    self.root.after(0, lambda: self._finish_stopped(result_text, "Batch captioning stopped."))
                else:
                    self.root.after(0, lambda: self._finish_success(result_text, "Batch caption generation completed successfully."))
            else:
                caption = self._generate_single_caption(payload["image_path"], payload, processor, model, device)
                if not caption:
                    caption = "(No caption was generated.)"

                caption = self._apply_caption_prefix(caption, payload["caption_prefix"], payload["caption_prefix_position"])

                save_message = self._maybe_save_caption(payload, caption)
                result_text = f"Caption:\n{caption}"
                if save_message:
                    result_text += f"\n\n{save_message}"
                self._debug_log("caption ready")

                if self.stop_requested.is_set():
                    self.root.after(0, lambda: self._finish_stopped(result_text))
                else:
                    self.root.after(0, lambda: self._finish_success(result_text))
        except Exception as exc:
            self._debug_log(f"error: {type(exc).__name__}: {exc}")
            error_text = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            self.root.after(0, lambda: self._finish_error(error_text))

    def _generate_single_caption(self, image_path: str, payload: dict, processor: Any, model: Any, device: str) -> str:
        self.root.after(0, self._start_caption_timer)
        image = Image.open(image_path).convert("RGB")
        self._debug_log(f"opened image: {image_path}")
        try:
            caption = self._run_caption_pipeline(payload, processor, model, device, image)
        finally:
            self.root.after(0, lambda: self._stop_caption_timer(keep_elapsed=True))
        return caption

    def _run_caption_pipeline(self, payload: dict, processor: Any, model: Any, device: str, image: Image.Image) -> str:
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner that writes clear, detailed descriptions.",
            },
            {
                "role": "user",
                "content": payload["prompt"],
            },
        ]

        convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[convo_string], images=[image], return_tensors="pt")
        inputs = inputs.to(device)
        self._debug_log("prepared model inputs")

        pixel_dtype = torch.float16 if device == "cuda" else torch.float32
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(pixel_dtype)

        max_new_tokens = max(1, int(payload["max_new_tokens"]))
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": payload["do_sample"],
            "use_cache": True,
            "repetition_penalty": payload["repetition_penalty"],
            "top_p": payload["top_p"],
            "top_k": payload["top_k"],
            "temperature": payload["temperature"],
            "stopping_criteria": StoppingCriteriaList([StopGenerationCriteria(self.stop_requested)]),
        }

        with torch.inference_mode():
            self._debug_log("starting generation")
            generate_ids = model.generate(**inputs, **generation_kwargs)[0]
        self._debug_log("generation completed")

        prompt_length = inputs["input_ids"].shape[1]
        caption_ids = generate_ids[prompt_length:]
        caption = processor.tokenizer.decode(
            caption_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        return caption

    def _generate_batch_captions(self, payload: dict, processor: Any, model: Any, device: str) -> tuple[str, str, bool]:
        folder = Path(payload["batch_image_folder"])
        save_folder = Path(payload["batch_save_folder"])
        save_folder.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            [
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
            ]
        )
        if not image_files:
            raise ValueError(f"No supported image files were found in: {folder}")

        resume_path = self._batch_resume_path(save_folder)
        config_signature = self._batch_config_signature(payload, folder, save_folder, image_files)
        resume_state = self._load_batch_resume_state(resume_path)
        completed_files: set[str] = set()

        if resume_state.get("config_signature") == config_signature:
            completed_files.update(str(name) for name in resume_state.get("completed_files", []))
            if completed_files:
                self._debug_log(f"resuming batch with {len(completed_files)} completed image(s)")
        elif resume_state:
            self._debug_log("existing batch resume data ignored because configuration changed")

        completed_files.update(
            image_file.name
            for image_file in image_files
            if self._batch_output_path(payload, image_file, save_folder).exists()
        )

        pending_files = [image_file for image_file in image_files if image_file.name not in completed_files]
        skipped_count = len(image_files) - len(pending_files)

        if not pending_files:
            summary = f"Nothing to do. {len(image_files)} image(s) already captioned."
            self._set_batch_progress_async(summary)
            self._set_current_image_async("already completed")
            return summary, summary, False

        batch_start = time.perf_counter()
        lines = [
            f"Batch mode: {len(image_files)} image(s)",
            f"Images folder: {folder}",
            f"Save folder: {save_folder}",
            f"Already completed: {skipped_count}",
            "",
        ]
        total_generation_time = 0.0
        stopped = False
        estimated_total_text = "Estimated total: calculating..."

        self._set_batch_progress_async(f"Batch progress: 0/{len(pending_files)} | {estimated_total_text}")
        if skipped_count:
            self._set_status_async(f"Resuming batch: {skipped_count} image(s) already completed.", self.accent2)

        for index, image_file in enumerate(pending_files, start=1):
            if self.stop_requested.is_set():
                stopped = True
                self._debug_log("batch stop detected before next image")
                self._set_batch_progress_async(f"Batch progress: {index - 1}/{len(pending_files)} | Stopped")
                break

            image_start = time.perf_counter()
            self._debug_log(f"processing {index}/{len(pending_files)}: {image_file.name}")
            self._set_preview_async(str(image_file))
            self._set_status_async(f"Processing {index}/{len(pending_files)}: {image_file.name}", self.accent)
            self._set_batch_progress_async(f"Batch progress: {index}/{len(pending_files)} | {image_file.name}")
            self._set_current_image_async(f"{index}/{len(pending_files)} - {image_file.name}")
            caption = self._generate_single_caption(str(image_file), payload, processor, model, device)
            if not caption:
                caption = "(No caption was generated.)"

            caption = self._apply_caption_prefix(caption, payload["caption_prefix"], payload["caption_prefix_position"])

            save_path = self._save_batch_caption(payload, image_file, caption, save_folder)
            elapsed = time.perf_counter() - image_start
            total_generation_time += elapsed

            if index == 1:
                estimated_total_seconds = elapsed * len(pending_files)
            else:
                average_seconds = total_generation_time / index
                estimated_total_seconds = average_seconds * len(pending_files)

            estimated_total_text = f"Estimated total: {self._format_elapsed(estimated_total_seconds).replace('⏱ ', '')}"
            self._set_batch_progress_async(f"Batch progress: {index}/{len(pending_files)} | {image_file.name} | {estimated_total_text}")

            lines.append(f"[{index}/{len(pending_files)}] {image_file.name} - {elapsed:.2f}s")
            lines.append(caption)
            if save_path:
                lines.append(f"Saved to: {save_path}")
            lines.append("")
            self._debug_log(f"finished {image_file.name} in {elapsed:.2f}s")
            self._set_status_async(f"Finished {index}/{len(pending_files)}: {image_file.name} | {estimated_total_text}", self.good)

            completed_files.add(image_file.name)
            self._write_batch_resume_state(
                resume_path,
                {
                    "config_signature": config_signature,
                    "folder": str(folder),
                    "save_folder": str(save_folder),
                    "image_files": [path.name for path in image_files],
                    "completed_files": sorted(completed_files),
                    "last_completed": image_file.name,
                    "updated_at": time.time(),
                },
            )

            if self.stop_requested.is_set():
                stopped = True
                self._debug_log("batch stop detected after current image")
                self._set_batch_progress_async(f"Batch progress: {index}/{len(pending_files)} | Stopped")
                break

        total_elapsed = time.perf_counter() - batch_start
        average_elapsed = total_generation_time / max(1, len(pending_files))
        summary = f"Total time: {total_elapsed:.2f}s | Average per processed image: {average_elapsed:.2f}s"
        if stopped:
            summary = f"Stopped. {summary}"
        lines.append(summary)
        self._set_batch_progress_async(summary)
        if stopped:
            self._set_current_image_async("stopped")
        else:
            try:
                resume_path.unlink()
            except Exception:
                pass
        return "\n".join(lines).strip(), summary, stopped

    def _load_model(self, model_name: str, hf_token: str = "") -> tuple[Any, Any, str]:
        cache_key = (model_name, hf_token)
        if cache_key in self.model_cache:
            self._debug_log("model cache hit")
            return self.model_cache[cache_key]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        def announce(text: str):
            self.root.after(0, lambda: self._set_status(text, self.accent))

        announce(f"Loading {model_name} on {device}...")
        self._debug_log(f"loading model: {model_name}")
        load_kwargs: dict[str, Any] = {}
        token = hf_token.strip()
        if token:
            load_kwargs["token"] = token

        processor: Any = AutoProcessor.from_pretrained(model_name, **load_kwargs)

        if device == "cuda":
            model: Any = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                dtype=dtype,
                device_map="auto",
                **load_kwargs,
            )
        else:
            model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                dtype=dtype,
                **load_kwargs,
            )
            model.to(torch.device(device))

        model.eval()
        cached = (processor, model, device)
        self.model_cache[cache_key] = cached
        self._debug_log("model loaded and cached")
        return cached

    def _clean_filename_component(self, value: str) -> str:
        cleaned = re.sub(r'[<>:"/\\|?*]+', "_", value.strip())
        cleaned = cleaned.replace(" ", "_")
        return cleaned.strip("._ ")

    def _batch_output_path(self, payload: dict, image_file: Path, save_folder: Path) -> Path:
        prefix = self._clean_filename_component(payload.get("batch_prefix", ""))
        base_name = image_file.stem if not prefix else f"{prefix}_{image_file.stem}"
        extension = ".json" if self.save_format_var.get() == "json" else ".txt"
        return save_folder / f"{base_name}{extension}"

    def _batch_resume_path(self, save_folder: Path) -> Path:
        return save_folder / ".caption_generator_resume.json"

    def _batch_config_signature(self, payload: dict, folder: Path, save_folder: Path, image_files: list[Path]) -> str:
        signature_payload = {
            "model_name": payload["model_name"],
            "prompt": payload["prompt"],
            "caption_prefix": payload["caption_prefix"],
            "caption_prefix_position": payload["caption_prefix_position"],
            "batch_prefix": payload["batch_prefix"],
            "save_format": self.save_format_var.get(),
            "max_new_tokens": payload["max_new_tokens"],
            "temperature": payload["temperature"],
            "top_p": payload["top_p"],
            "top_k": payload["top_k"],
            "repetition_penalty": payload["repetition_penalty"],
            "do_sample": payload["do_sample"],
            "folder": str(folder.resolve()),
            "save_folder": str(save_folder.resolve()),
            "images": [path.name for path in image_files],
        }
        digest = json.dumps(signature_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(digest).hexdigest()

    def _load_batch_resume_state(self, resume_path: Path) -> dict[str, Any]:
        if not resume_path.exists():
            return {}
        try:
            return json.loads(resume_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_batch_resume_state(self, resume_path: Path, state: dict[str, Any]):
        try:
            resume_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            self._debug_log("failed to update batch resume state")

    def _apply_caption_prefix(self, caption: str, prefix: str, position: str) -> str:
        caption = caption.strip()
        prefix = prefix.strip()
        if not prefix:
            return caption
        if not caption:
            return prefix
        if position == "last":
            return f"{caption}, {prefix}"
        return f"{prefix}, {caption}"

    def _maybe_save_caption(self, payload: dict, caption: str) -> str:
        save_path = payload["save_path"]
        if not save_path:
            return ""

        path = Path(save_path)
        suffix = path.suffix.lower()
        if suffix not in {".txt", ".json"}:
            suffix = ".json" if self.save_format_var.get() == "json" else ".txt"
            path = path.with_suffix(suffix)

        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".json":
            data = {
                "model_name": payload["model_name"],
                "image_path": payload["image_path"],
                "prompt": payload["prompt"],
                "caption": caption,
                "generation": {
                    "max_new_tokens": payload["max_new_tokens"],
                    "temperature": payload["temperature"],
                    "top_p": payload["top_p"],
                    "top_k": payload["top_k"],
                    "repetition_penalty": payload["repetition_penalty"],
                    "do_sample": payload["do_sample"],
                },
            }
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            path.write_text(caption.strip() + "\n", encoding="utf-8")

        return f"Saved to: {path}"

    def _save_batch_caption(self, payload: dict, image_file: Path, caption: str, save_folder: Path) -> str:
        path = self._batch_output_path(payload, image_file, save_folder)

        if path.suffix.lower() == ".json":
            data = {
                "image_path": str(image_file),
                "caption": caption,
                "generation": {
                    "max_new_tokens": payload["max_new_tokens"],
                    "temperature": payload["temperature"],
                    "top_p": payload["top_p"],
                    "top_k": payload["top_k"],
                    "repetition_penalty": payload["repetition_penalty"],
                    "do_sample": payload["do_sample"],
                },
            }
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            path.write_text(caption.strip() + "\n", encoding="utf-8")

        return str(path)

    def _finish_success(self, result_text: str, status_text: str = "Caption generated successfully."):
        self._reset_caption_timer()
        self._set_output(result_text)
        self._set_status(status_text, self.good)
        self.generate_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.current_worker = None
        messagebox.showinfo("Done", status_text)

    def _finish_stopped(self, result_text: str, status_text: str = "Captioning stopped."):
        self._reset_caption_timer()
        self._set_output(result_text)
        self._set_status(status_text, self.accent2)
        self.generate_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.current_worker = None
        messagebox.showinfo("Stopped", status_text)

    def _finish_error(self, error_text: str):
        self._reset_caption_timer()
        self._set_output(f"Error:\n{error_text}")
        self._set_status("Generation failed. Check the output for details.", self.error)
        self.generate_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.current_worker = None
        messagebox.showerror("Generation error", error_text.splitlines()[0])


def main():
    root = tk.Tk()
    app = CaptionGeneratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
