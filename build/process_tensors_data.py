import os
import glob
from collections import defaultdict

def process_files(directory):
    # kernel -> { tensor_name: [taco_val, atl_val, taco_val/atl_val] }
    data = defaultdict(dict)

    for filepath in glob.glob(os.path.join(directory, "*.txt")):
        filename = os.path.basename(filepath)
        kernel_name = filename.replace(".txt", "")
        if (kernel_name != "mttkrp" and kernel_name != "ttv"):
            continue
        
        with open(filepath, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        i = 0
        while i < len(lines):
            header = lines[i]
            parts = header.split()
            system = parts[0]         # "TACO" or "ATL"
            if (system == "TACO" or system == "ATL"):
                tensor_name = parts[-1]
                tensor_name = "".join(char for char in tensor_name if char.isalnum())
                value = float(lines[i + 1])
                if tensor_name not in data[kernel_name]:
                    data[kernel_name][tensor_name] = [None, None]

                if system == "TACO":
                    data[kernel_name][tensor_name][0] = value
                elif system == "ATL":
                    data[kernel_name][tensor_name][1] = value

                i += 2
            else:
                i += 1

    result = {}

    for kernel_name, tensor_map in data.items():
        result[kernel_name] = {}
        for tensor, vals in tensor_map.items():
            taco, atl = vals

            if taco is None or atl is None:
                raise ValueError(
                    f"Incomplete data for kernel='{kernel_name}', tensor='{tensor}': {vals}"
                )

            if atl == 0:
                ratio = float("inf")  # or raise an error if you prefer
            else:
                ratio = taco / atl

            result[kernel_name][tensor] = (taco, atl, ratio)

    return result

if __name__ == "__main__":
    directory = "./"  # change this
    result = process_files(directory)

    for fmt, tensor_map in result.items():
        print(f"Kernel: {fmt}")
        for tensor, values in tensor_map.items():
            print(f"  {tensor}: {values}")
