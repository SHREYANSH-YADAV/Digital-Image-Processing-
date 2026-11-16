"""
CLAHE Satellite Image Enhancer (with GUI)
-----------------------------------------
Requirements:  pip install opencv-python numpy pillow
Run:           python clahe_satellite_gui.py

Features
  * Opens PNG / JPG / TIFF (8-bit or 16-bit) satellite images
  * CLAHE on luminance only (LAB / HSV) -> boosts contrast without colour shifts
  * Live side-by-side preview (computed on a downscaled copy for speed)
  * Full-resolution processing + save runs in a background thread
"""

import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

PREVIEW_MAX = 1600          # longest side (px) used for live preview
DEBOUNCE_MS = 120           # delay before re-computing after slider move
MODES = ["LAB (L channel)", "HSV (V channel)", "RGB (per channel)", "Grayscale"]


# --------------------------------------------------------------------------- #
#  Image helpers
# --------------------------------------------------------------------------- #
def read_image(path: str) -> np.ndarray:
    """Unicode-safe read that keeps bit depth (needed for 16-bit GeoTIFFs)."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Unsupported or corrupted image file.")
    if img.ndim == 3:
        if img.shape[2] == 4:       # drop alpha
            img = img[:, :, :3]
        elif img.shape[2] == 2:
            img = img[:, :, 0]
    return img


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert 16-bit / float imagery to 8-bit using a 0.5-99.5 % stretch."""
    if img.dtype == np.uint8:
        return img
    f = img.astype(np.float32)
    lo, hi = np.percentile(f[::4, ::4], (0.5, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((f - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def apply_clahe(img8: np.ndarray, clip: float, tile: int, mode: str) -> np.ndarray:
    """Apply CLAHE to an 8-bit image (BGR or gray) and return the same layout."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))

    if img8.ndim == 2 or mode == MODES[3]:
        gray = img8 if img8.ndim == 2 else cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
        return clahe.apply(gray)

    if mode == MODES[0]:                                   # LAB
        lab = cv2.cvtColor(img8, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if mode == MODES[1]:                                   # HSV
        hsv = cv2.cvtColor(img8, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return cv2.merge([clahe.apply(c) for c in cv2.split(img8)])   # per-channel


def to_pil_fit(img8: np.ndarray, w: int, h: int) -> Image.Image:
    """Resize (keeping aspect) to fit w x h and convert to a PIL RGB image."""
    ih, iw = img8.shape[:2]
    s = min(w / iw, h / ih)
    if s < 1:
        img8 = cv2.resize(img8, (max(1, int(iw * s)), max(1, int(ih * s))),
                          interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB if img8.ndim == 2 else cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# --------------------------------------------------------------------------- #
#  GUI
# --------------------------------------------------------------------------- #
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CLAHE Satellite Image Enhancer")
        self.geometry("1280x760")
        self.minsize(900, 560)

        self.src_full = None      # 8-bit full-res source
        self.src_prev = None      # downscaled preview source
        self.enh_prev = None      # enhanced preview
        self.path = None
        self._job = None
        self._photos = [None, None]

        self._build_ui()

    # ---- UI -------------------------------------------------------------- #
    def _build_ui(self):
        bar = ttk.Frame(self, padding=8)
        bar.pack(fill="x")

        ttk.Button(bar, text="Open Image…", command=self.open_image).pack(side="left")
        ttk.Button(bar, text="Save Result…", command=self.save_image).pack(side="left", padx=6)
        ttk.Button(bar, text="Reset", command=self.reset).pack(side="left")

        ttk.Label(bar, text="  Mode:").pack(side="left", padx=(16, 4))
        self.mode = tk.StringVar(value=MODES[0])
        cb = ttk.Combobox(bar, textvariable=self.mode, values=MODES, state="readonly", width=18)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self.schedule())

        sl = ttk.Frame(self, padding=(8, 0))
        sl.pack(fill="x")
        self.clip = tk.DoubleVar(value=2.0)
        self.tile = tk.IntVar(value=8)
        self._slider(sl, "Clip limit", self.clip, 1.0, 10.0, 0.1, 0)
        self._slider(sl, "Tile grid size", self.tile, 2, 32, 1, 1)
        sl.columnconfigure(1, weight=1)

        panes = ttk.Frame(self, padding=8)
        panes.pack(fill="both", expand=True)
        panes.columnconfigure((0, 1), weight=1, uniform="p")
        panes.rowconfigure(1, weight=1)

        for i, t in enumerate(("Original", "CLAHE Enhanced")):
            ttk.Label(panes, text=t, font=("Segoe UI", 11, "bold")).grid(row=0, column=i, pady=(0, 4))
        self.canv = []
        for i in range(2):
            c = tk.Canvas(panes, bg="#1e1e1e", highlightthickness=0)
            c.grid(row=1, column=i, sticky="nsew", padx=4)
            c.bind("<Configure>", lambda e: self.render())
            self.canv.append(c)

        self.status = tk.StringVar(value="Open a satellite image to begin.")
        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w",
                  padding=4).pack(fill="x", side="bottom")

    def _slider(self, parent, text, var, lo, hi, step, row):
        ttk.Label(parent, text=text, width=14).grid(row=row, column=0, sticky="w")
        s = tk.Scale(parent, variable=var, from_=lo, to=hi, resolution=step,
                     orient="horizontal", showvalue=True, command=lambda v: self.schedule())
        s.grid(row=row, column=1, sticky="ew", padx=6)

    # ---- actions --------------------------------------------------------- #
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Select satellite image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All files", "*.*")])
        if not path:
            return
        try:
            raw = read_image(path)
            self.src_full = to_uint8(raw)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image:\n{e}")
            return

        self.path = path
        h, w = self.src_full.shape[:2]
        s = min(1.0, PREVIEW_MAX / max(h, w))
        self.src_prev = (self.src_full if s == 1.0 else
                         cv2.resize(self.src_full, (int(w * s), int(h * s)),
                                    interpolation=cv2.INTER_AREA))
        self.status.set(f"Loaded: {path}   |   {w}×{h}px   |   original dtype: {raw.dtype}")
        self.recompute()

    def reset(self):
        self.clip.set(2.0)
        self.tile.set(8)
        self.mode.set(MODES[0])
        self.schedule()

    def schedule(self):
        """Debounce slider movement so we don't recompute on every pixel."""
        if self.src_prev is None:
            return
        if self._job:
            self.after_cancel(self._job)
        self._job = self.after(DEBOUNCE_MS, self.recompute)

    def recompute(self):
        self._job = None
        if self.src_prev is None:
            return
        self.enh_prev = apply_clahe(self.src_prev, self.clip.get(), self.tile.get(), self.mode.get())
        self.render()

    def render(self):
        if self.src_prev is None or self.enh_prev is None:
            return
        for i, img in enumerate((self.src_prev, self.enh_prev)):
            c = self.canv[i]
            w, h = max(c.winfo_width(), 50), max(c.winfo_height(), 50)
            self._photos[i] = ImageTk.PhotoImage(to_pil_fit(img, w, h))
            c.delete("all")
            c.create_image(w // 2, h // 2, image=self._photos[i], anchor="center")

    def save_image(self):
        if self.src_full is None:
            messagebox.showinfo("Nothing to save", "Open an image first.")
            return
        out = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("TIFF", "*.tif")])
        if not out:
            return
        clip, tile, mode = self.clip.get(), self.tile.get(), self.mode.get()
        self.status.set("Processing full-resolution image…")

        def work():
            try:
                res = apply_clahe(self.src_full, clip, tile, mode)
                ok, buf = cv2.imencode("." + out.rsplit(".", 1)[-1].lower(), res)
                if not ok:
                    raise RuntimeError("Encoding failed.")
                buf.tofile(out)
                self.after(0, lambda: self.status.set(f"Saved: {out}"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Save error", str(e)))

        threading.Thread(target=work, daemon=True).start()


if __name__ == "__main__":
    App().mainloop()