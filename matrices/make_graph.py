import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def process_files(directory):

    for filepath in glob.glob(os.path.join(directory, "*.csv")):

        filename = os.path.basename(filepath)
        format_name = filename.replace(".csv", "")
        df = pd.read_csv(filename)
        df = df.sort_values(by="Matrix")
        # Extract columns
        matrices = df["Matrix"]
        speedup = df[" TACO/ATL"]

        # Plot
        plt.figure(figsize=(3.32, 2.82))
        plt.grid(axis="y", linestyle="-", linewidth=0.5, color=(217/255,217/255,217/255))
        plt.bar(matrices, speedup, color=(0, 112/255, 192/255), width=0.3)
        ax = plt.gca()
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)
        plt.ylabel("Relative Throughput",fontsize=10)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks([0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2],fontsize=8)
        plt.axhline(1.0, linestyle="-",color=(0, 176/255, 80/255))
        plt.title(format_name, fontsize=14)
        

        plt.tight_layout()
        plt.savefig(format_name + ".pdf")


if __name__ == "__main__":
    directory = "./"  # change this
    result = process_files(directory)
