import sys
import heapq
import cv2
import numpy as np
from collections import Counter


class HuffmanNode:
    def __init__(self, symbol, freq, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    # heapq needs a comparison operator; break ties using an incrementing id
    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(freq_counter):
    heap = []
    uid = 0
    for symbol, freq in freq_counter.items():
        heapq.heappush(heap, (freq, uid, HuffmanNode(symbol, freq)))
        uid += 1

    if len(heap) == 1:
        # Edge case: only one distinct symbol in the whole image
        _, _, only_node = heap[0]
        return HuffmanNode(None, only_node.freq, left=only_node)

    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        merged = HuffmanNode(None, f1 + f2, left=n1, right=n2)
        heapq.heappush(heap, (merged.freq, uid, merged))
        uid += 1

    return heap[0][2]


def build_codes(node, prefix="", codes=None):
    if codes is None:
        codes = {}
    if node is None:
        return codes
    if node.symbol is not None:
        codes[node.symbol] = prefix if prefix else "0"
        return codes
    build_codes(node.left, prefix + "0", codes)
    build_codes(node.right, prefix + "1", codes)
    return codes


def analyze_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pixels = gray.flatten()
    total = len(pixels)

    freq_counter = Counter(pixels.tolist())
    tree_root = build_huffman_tree(freq_counter)
    codes = build_codes(tree_root)

    probs = {sym: f / total for sym, f in freq_counter.items()}
    entropy = -sum(p * np.log2(p) for p in probs.values())
    avg_len = sum(probs[sym] * len(codes[sym]) for sym in freq_counter)
    efficiency = (entropy / avg_len) * 100 if avg_len > 0 else 0
    compression_ratio = 8 / avg_len if avg_len > 0 else 0

    symbols_freqs = sorted(freq_counter.items(), key=lambda x: x[1], reverse=True)
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
    for sym, freq in symbols_freqs[:30]:
        print(f"{sym:>8} {freq:>10} {probs[sym]:>10.5f} {codes[sym]:>15} {len(codes[sym]):>10}")
    if len(symbols_freqs) > 30:
        print(f"... ({len(symbols_freqs) - 30} more symbols not shown)")

    print("\n--- Summary ---")
    print(f"Number of distinct symbols : {len(symbols_freqs)}")
    print(f"Entropy                    : {entropy:.4f} bits/symbol")
    print(f"Average Huffman length     : {avg_len:.4f} bits/symbol")
    print(f"Coding Efficiency          : {efficiency:.2f}%")
    print(f"Compression Ratio (vs 8bit): {ratio:.3f} : 1")


if __name__ == "__main__":
    main()