import sys
import cv2
import numpy as np
from collections import Counter


class SFNode:
    __slots__ = ("symbol", "freq")
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq


def shannon_fano(symbols_freqs):
    codes = {s: "" for s, _ in symbols_freqs}

    def split(lst):
        if len(lst) <= 1:
            return
        total = sum(f for _, f in lst)
        acc = 0
        split_idx = 0
        best_diff = float("inf")
        # find split point that minimizes |left_sum - right_sum|
        for i in range(len(lst)):
            acc += lst[i][1]
            diff = abs((total - acc) - acc)
            if diff < best_diff:
                best_diff = diff
                split_idx = i
        left = lst[: split_idx + 1]
        right = lst[split_idx + 1 :]
        for sym, _ in left:
            codes[sym] += "0"
        for sym, _ in right:
            codes[sym] += "1"
        split(left)
        split(right)

    split(symbols_freqs)
    return codes


def analyze_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pixels = gray.flatten()
    total = len(pixels)

    freq_counter = Counter(pixels.tolist())
    symbols_freqs = sorted(freq_counter.items(), key=lambda x: x[1], reverse=True)

    codes = shannon_fano(symbols_freqs)

    probs = {sym: f / total for sym, f in symbols_freqs}
    entropy = -sum(p * np.log2(p) for p in probs.values())
    avg_len = sum(probs[sym] * len(codes[sym]) for sym, _ in symbols_freqs)
    efficiency = (entropy / avg_len) * 100 if avg_len > 0 else 0
    compression_ratio = 8 / avg_len if avg_len > 0 else 0

    return symbols_freqs, codes, probs, entropy, avg_len, efficiency, compression_ratio


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(title="Select an image")
        if not path:
            print("No file selected.")
            return

    symbols_freqs, codes, probs, entropy, avg_len, efficiency, ratio = analyze_image(path)

    print(f"{'Symbol':>8} {'Freq':>10} {'Prob':>10} {'Code':>15} {'Code Len':>10}")
    print("-" * 60)
    for sym, freq in symbols_freqs[:30]:  # print first 30 rows to keep it readable
        print(f"{sym:>8} {freq:>10} {probs[sym]:>10.5f} {codes[sym]:>15} {len(codes[sym]):>10}")
    if len(symbols_freqs) > 30:
        print(f"... ({len(symbols_freqs) - 30} more symbols not shown)")

    print("\n--- Summary ---")
    print(f"Number of distinct symbols : {len(symbols_freqs)}")
    print(f"Entropy                    : {entropy:.4f} bits/symbol")
    print(f"Average Shannon-Fano length: {avg_len:.4f} bits/symbol")
    print(f"Coding Efficiency          : {efficiency:.2f}%")
    print(f"Compression Ratio (vs 8bit): {ratio:.3f} : 1")


if __name__ == "__main__":
    main()