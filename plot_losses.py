#!/usr/bin/env python3
import argparse
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_file(file_path):
    """
    Parse a log file to extract step, train loss, val loss, and elapsed time.

    Args:
        file_path: Path to the log file

    Returns:
        Dictionary containing parsed data
    """
    steps = []
    train_losses = []
    val_losses = []
    elapsed_times_dict = {}  # Use a dictionary to map step to elapsed time

    # Regular expressions for matching the relevant lines
    step_pattern = re.compile(r'step (\d+): train loss ([\d\.]+), val loss ([\d\.]+)')
    elapsed_pattern = re.compile(r'(\d+) \| next_token_loss [\d\.]+ \| full_loss [\d\.]+ \| lr [\d\.\-e]+ \| elapsed ([\d\.]+)s')

    with open(file_path, 'r') as f:
        for line in f:
            # Match step lines with train and val loss
            step_match = step_pattern.search(line)
            if step_match:
                step = int(step_match.group(1))
                train_loss = float(step_match.group(2))
                val_loss = float(step_match.group(3))

                steps.append(step)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

            # Match lines with elapsed time
            elapsed_match = elapsed_pattern.search(line)
            if elapsed_match:
                iteration = int(elapsed_match.group(1))
                elapsed = float(elapsed_match.group(2))

                # Store elapsed time for each iteration
                elapsed_times_dict[iteration] = elapsed

    # Create elapsed_times array that matches steps
    elapsed_times = []
    for step in steps:
        if step in elapsed_times_dict:
            elapsed_times.append(elapsed_times_dict[step])
        else:
            # If we don't have elapsed time for this step, use the previous one or 0
            elapsed_times.append(elapsed_times[-1] if elapsed_times else 0)

    return {
        'steps': np.array(steps),
        'train_losses': np.array(train_losses),
        'val_losses': np.array(val_losses),
        'elapsed_times': np.array(elapsed_times)
    }

def plot_losses(log_files, figsize=(12, 8)):
    """
    Plot train and validation losses from multiple log files.

    Args:
        log_files: List of paths to log files
        figsize: Size of the figure (width, height) in inches
    """
    # Create figure with two subplots (step-based and time-based)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Define a list of colors for different log files
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for i, log_file in enumerate(log_files):
        file_name = Path(log_file).stem
        color = colors[i % len(colors)]

        # Parse the log file
        data = parse_log_file(log_file)

        # Plot step-based losses
        axes[0].plot(data['steps'], data['train_losses'], '-o', color=color, label=f'{file_name} - Train Loss')
        axes[0].plot(data['steps'], data['val_losses'], '--s', color=color, alpha=0.7, label=f'{file_name} - Val Loss')

        # Plot time-based losses
        if len(data['elapsed_times']) > 0:
            axes[1].plot(data['elapsed_times'], data['train_losses'], '-o', color=color, label=f'{file_name} - Train Loss')
            axes[1].plot(data['elapsed_times'], data['val_losses'], '--s', color=color, alpha=0.7, label=f'{file_name} - Val Loss')

    # Configure step-based plot
    axes[0].set_xlabel('Step', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss vs. Step', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    # Configure time-based plot
    axes[1].set_xlabel('Elapsed Time (s)', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Training and Validation Loss vs. Time', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig('loss_plots.png', dpi=300)
    print(f"Plots saved to loss_plots.png")

    # Show the plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot training and validation losses from log files.')
    parser.add_argument('log_files', nargs='+', help='Path(s) to log file(s)')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 8], help='Figure size (width height) in inches')

    args = parser.parse_args()

    plot_losses(args.log_files, figsize=tuple(args.figsize))

if __name__ == '__main__':
    main()
