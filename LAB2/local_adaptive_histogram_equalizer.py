import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
import numpy as np


class ColorHistogramEqualizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Histogram Equalization Dashboard")
        self.root.geometry("1400x900")

        # Force window to front
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(1000, lambda: self.root.attributes("-topmost", False))

        self.cap = None
        self.current_frame = None  # BGR uint8 frame

        # --- Controls ---
        control_frame = tk.Frame(root, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_start = tk.Button(
            control_frame, text="Start Camera", command=self.start_camera,
            bg="#4CAF50", fg="white", font=("Arial", 11, "bold"),
        )
        self.btn_start.pack(side=tk.LEFT, padx=15)

        self.btn_capture = tk.Button(
            control_frame, text="Capture Photo", command=self.capture_photo,
            bg="#2196F3", fg="white", font=("Arial", 11, "bold"), state=tk.DISABLED,
        )
        self.btn_capture.pack(side=tk.LEFT, padx=15)

        self.btn_upload = tk.Button(
            control_frame, text="Upload Image", command=self.upload_image,
            bg="#FF9800", fg="white", font=("Arial", 11, "bold"),
        )
        self.btn_upload.pack(side=tk.LEFT, padx=15)

        # --- Main Grid: 2 columns (Original vs Equalized) x 2 rows (Image, Gaussian Hist) ---
        self.grid_frame = tk.Frame(root, bg="gray15")
        self.grid_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        for col in range(2):
            self.grid_frame.columnconfigure(col, weight=1)
        self.grid_frame.rowconfigure(1, weight=1)
        self.grid_frame.rowconfigure(2, weight=1)

        titles = ["Original (Colorful)", "Equalized (Colorful)"]
        self.image_labels = []
        self.hist_labels = []

        for col in range(2):
            lbl_title = tk.Label(
                self.grid_frame, text=titles[col], font=("Arial", 13, "bold"),
                bg="gray15", fg="white",
            )
            lbl_title.grid(row=0, column=col, padx=5, pady=5)

            img_lbl = tk.Label(
                self.grid_frame, text="No Image", bg="gray30", fg="white",
                width=55, height=18,
            )
            img_lbl.grid(row=1, column=col, padx=8, pady=5, sticky="nsew")
            self.image_labels.append(img_lbl)

            hist_lbl = tk.Label(
                self.grid_frame, text="No Histogram", bg="lightgray", fg="black",
                width=55, height=18,
            )
            hist_lbl.grid(row=2, column=col, padx=8, pady=5, sticky="nsew")
            self.hist_labels.append(hist_lbl)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ------------------------------------------------------------------ #
    # Camera handling
    # ------------------------------------------------------------------ #
    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror(
                    "Error", "Could not open webcam. Try uploading an image instead."
                )
                self.cap = None
                return
            self.btn_capture.config(state=tk.NORMAL)
            self.btn_start.config(text="Stop Camera", bg="#f44336")
            self.show_camera_feed()
        else:
            self.stop_camera()

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.btn_capture.config(state=tk.DISABLED)
        self.btn_start.config(text="Start Camera", bg="#4CAF50")

    def show_camera_feed(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = cv2.flip(frame, 1)
                self.update_image_previews_only()
            self.root.after(30, self.show_camera_feed)

    def capture_photo(self):
        if self.current_frame is not None:
            temp_frame = self.current_frame.copy()
            self.stop_camera()
            self.current_frame = temp_frame
            self.update_all_displays()
            messagebox.showinfo("Success", "Photo captured successfully!")

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.stop_camera()
            img = cv2.imread(file_path)
            if img is not None:
                self.current_frame = cv2.resize(img, (640, 480))
                self.update_all_displays()
            else:
                messagebox.showerror("Error", "Failed to load the image.")

    # ------------------------------------------------------------------ #
    # Core processing: Global Histogram Equalization
    # ------------------------------------------------------------------ #
    def apply_histogram_equalization(self, bgr_img):
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_eq = cv2.equalizeHist(l_channel)
        lab_eq = cv2.merge([l_eq, a_channel, b_channel])
        bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        return bgr_eq

    # ------------------------------------------------------------------ #
    # Histogram plotting
    # ------------------------------------------------------------------ #
    def make_histogram_plot(self, bgr_img, title):
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        x_vals = np.arange(0, 256)

        fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=100)
        ax.bar(x_vals, hist, width=1.0, color="steelblue", edgecolor="none")

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Pixel Intensity", fontsize=8)
        ax.set_ylabel("Pixel Count", fontsize=8)
        ax.set_xlim([0, 256])
        ax.tick_params(axis="both", which="major", labelsize=7)
        fig.tight_layout()

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()
        hist_img = Image.frombytes("RGBA", (width, height), s).resize((420, 300))
        plt.close(fig)
        return hist_img

    # ------------------------------------------------------------------ #
    # Display helpers
    # ------------------------------------------------------------------ #
    def update_image_previews_only(self):
        if self.current_frame is None:
            return

        original_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        equalized_bgr = self.apply_histogram_equalization(self.current_frame)
        equalized_rgb = cv2.cvtColor(equalized_bgr, cv2.COLOR_BGR2RGB)

        for lbl, img_rgb in zip(self.image_labels, [original_rgb, equalized_rgb]):
            img_pil = Image.fromarray(img_rgb).resize((420, 300))
            imgtk = ImageTk.PhotoImage(image=img_pil)
            lbl.imgtk = imgtk
            lbl.config(image=imgtk, text="")

    def update_all_displays(self):
        if self.current_frame is None:
            return

        original_bgr = self.current_frame
        equalized_bgr = self.apply_histogram_equalization(original_bgr)

        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        equalized_rgb = cv2.cvtColor(equalized_bgr, cv2.COLOR_BGR2RGB)

        # Update the two image previews
        for lbl, img_rgb in zip(self.image_labels, [original_rgb, equalized_rgb]):
            img_pil = Image.fromarray(img_rgb).resize((420, 300))
            imgtk = ImageTk.PhotoImage(image=img_pil)
            lbl.imgtk = imgtk
            lbl.config(image=imgtk, text="")

        # Update the two histogram plots
        plots = [
            (original_bgr, "Histogram - Original Colorful Image"),
            (equalized_bgr, "Histogram - Equalized Image"),
        ]
        for lbl, (img_bgr, title) in zip(self.hist_labels, plots):
            hist_img = self.make_histogram_plot(img_bgr, title)
            hist_imgtk = ImageTk.PhotoImage(image=hist_img)
            lbl.imgtk = hist_imgtk
            lbl.config(image=hist_imgtk, text="")

    def on_closing(self):
        self.stop_camera()
        self.root.destroy()
        sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = ColorHistogramEqualizationApp(root)
    root.mainloop()