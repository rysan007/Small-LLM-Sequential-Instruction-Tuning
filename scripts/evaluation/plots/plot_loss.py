import re
import argparse
import matplotlib.pyplot as plt

def plot_loss(log_file_path, output_image_path, title):
    epochs = []
    losses = []

    # Updated Regex to handle optional quotes around the numbers!
    # Matches both {'loss': 1.234} AND {'loss': '1.234'}
    regex = r"'loss':\s*'?([0-9.]+)'?.*?'epoch':\s*'?([0-9.]+)'?"

    print(f"Scanning {log_file_path} for training loss data...")
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(regex, line)
            if match:
                loss = float(match.group(1))
                epoch = float(match.group(2))
                losses.append(loss)
                epochs.append(epoch)

    if not losses:
        print("Error: Could not find any training loss data in the log file.")
        print("Ensure this is the correct SLURM .out file containing Hugging Face logs.")
        return

    print(f"Found {len(losses)} logging steps. Generating plot...")

    # Set up the plot style for a professional look
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='#1f77b4', markersize=4, linewidth=2)
    
    # Add titles and labels
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save and close
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    
    print(f"Success! Loss curve saved to {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, required=True, help="Path to your SLURM .out file")
    parser.add_argument("--output", type=str, default="loss_curve.png", help="Output PNG file name")
    parser.add_argument("--title", type=str, default="Stage 1: Alpaca Training Loss", help="Title for the chart")
    args = parser.parse_args()

    plot_loss(args.log_file, args.output, args.title)