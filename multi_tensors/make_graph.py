import os
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def process_files(directory):
    colors = [(55/255,88/255,153/255),(63/255,106/255,183/255),(121/255,145/255,206/255),(179/255,190/255,223/255)]
    fig,ax = plt.subplots(figsize=(5, 3.32))
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.grid(axis="y",linestyle="-",linewidth=0.5,color=(217/255,217/255,217/255))
    ax.tick_params(axis="both", which="both", length=0)
    plt.yticks([0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2],fontsize=8)
    plt.axhline(1.0, linestyle="-",color=(0, 176/255, 80/255))
    ax.set_ylim(0, 2.0)
    
    all_values = []
    matrices = []
    group_names = []
    
    for filepath in glob.glob(os.path.join(directory, "*.csv")):
        filename = os.path.basename(filepath)
        format_name = filename.replace(".csv", "")
        group_names.append(format_name)
        df = pd.read_csv(filename)
        # df = df.sort_values(by="Matrix")
        # Extract columns
        matrices = df["Sparsity"].astype(str)
        all_values.append(df[" TACO/ATL"])
    
        # Plot
        #for i in range(len(speedup)):
        #    plt.bar(matrices, speedup, color=(0, 112/255, 192/255), width=0.3)

    values = np.array(all_values).T
    num_series, num_groups = values.shape
    x = np.arange(num_groups)

    group_width = 0.6
    slot_width = group_width / num_series
    bar_width = slot_width * 0.8   # shrink → creates gaps

    for i in range(num_series):
        ax.bar(
            x - group_width/2 + i * slot_width + slot_width/2,
            values[i],
            width=bar_width,
            label=matrices[i],
            color=colors[i % len(colors)],
            # edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(group_names)
    ax.legend(loc="center right", bbox_to_anchor=(1.02, 1), frameon=False)
    plt.subplots_adjust(left=0.75)

    ax.set_ylabel("Relative Throughput")
    ax.set_xlabel("Sparse Matrix")
    
    ax.legend(frameon=False)
    
    plt.tight_layout()
    
    plt.title("Sparse-Tensor Times Sparse Vector", fontsize=12)
    plt.savefig("spttspv.pdf",bbox_inches="tight",pad_inches=0.1)

if __name__ == "__main__":
    directory = "./"  # change this
    result = process_files(directory)
