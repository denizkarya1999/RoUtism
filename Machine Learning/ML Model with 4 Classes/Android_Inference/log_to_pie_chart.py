#!/usr/bin/env python3

import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Constants
LOG_FILE = 'RoEmotion_Predictions_Log.txt'
CLASSES = ["Angry", "Excitement", "Anxious", "Sadness"]
RESULTS_DIR = 'results'

def main():
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Data structures to collect counts and total accuracy per class
    emotion_counts = {cls: 0 for cls in CLASSES}
    emotion_acc_sums = {cls: 0.0 for cls in CLASSES}

    # Regex to match lines like:
    # 2025-05-31 03:36:14 => Excitement (79.6%)
    pattern = re.compile(r'.*=>\s*(\w+)\s*\((\d+\.\d+)%\)')

    # Parse the log file
    with open(LOG_FILE, 'r') as f:  # :contentReference[oaicite:0]{index=0}
        for line in f:
            line = line.strip()
            m = pattern.match(line)
            if not m:
                continue

            emotion = m.group(1)
            acc_pct = float(m.group(2)) / 100.0  # Convert percent to a fraction

            if emotion not in CLASSES:
                # Skip any unexpected emotion labels
                continue

            emotion_counts[emotion] += 1
            emotion_acc_sums[emotion] += acc_pct

    # Compute average accuracy per emotion
    emotion_avg_acc = {}
    for cls in CLASSES:
        count = emotion_counts[cls]
        if count > 0:
            emotion_avg_acc[cls] = emotion_acc_sums[cls] / count
        else:
            emotion_avg_acc[cls] = 0.0

    # Print summary to console
    print("Emotion Summary (based on RoEmotion_Predictions_Log.txt):\n")
    for cls in CLASSES:
        print(f"{cls}:")
        print(f"  - Times recorded: {emotion_counts[cls]}")
        print(f"  - Average accuracy: {emotion_avg_acc[cls]:.4f}\n")

    # Create a pie chart of emotion counts
    counts = [emotion_counts[cls] for cls in CLASSES]
    total = sum(counts)

    if total == 0:
        print("No emotion entries found in the log. Skipping pie chart generation.")
        return

    # Filter out zero-count emotions so the pie chart does not include them
    labels_nonzero = []
    counts_nonzero = []
    for idx, cls in enumerate(CLASSES):
        if counts[idx] > 0:
            labels_nonzero.append(cls)
            counts_nonzero.append(counts[idx])

    # Assign colors for each emotion (matching the order in CLASSES)
    color_map = {
        "Angry": "#FF9999",
        "Excitement": "#66B3FF",
        "Anxious": "#99FF99",
        "Sadness": "#FFCC99"
    }
    colors_nonzero = [color_map[cls] for cls in labels_nonzero]

    plt.figure(figsize=(6, 6))
    plt.pie(
        counts_nonzero,
        labels=labels_nonzero,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct * total / 100))})",
        startangle=140,
        colors=colors_nonzero,
        wedgeprops={'alpha': 0.8}
    )
    plt.title("Emotion Distribution (Counts)")

    # Save the pie chart
    pie_path = os.path.join(RESULTS_DIR, 'emotion_counts_pie.png')
    plt.tight_layout()
    plt.savefig(pie_path, dpi=300)
    plt.close()

    print(f"Pie chart saved to: {pie_path}")

if __name__ == "__main__":
    main()
