import os
import glob
from collections import defaultdict

def process_files(directory):
    # kernel -> { tensor_name: [taco_val, atl_val, taco_val/atl_val] }
    data = defaultdict(dict)

    for filepath in glob.glob(os.path.join(directory, "*.txt")):
        filename = os.path.basename(filepath)
        tensor_name = filename.replace(".tns.txt", "")
        
        with open(filepath, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        i = 0
        while i < len(lines):
            header = lines[i]
            parts = header.split()
            system = parts[0]
            if (system == "TACO" or system == "ATL"):
                tensor_sparsity = parts[-1]
                tensor_sparsity = "".join(char for char in tensor_sparsity if char != ':')
                
                value_line = lines[i+1]
                value_parts = value_line.split()
                if (1 < len(value_parts)):
                    i = i+1
                    value_line = lines[i+1]
                value = float(value_line)
                
                if tensor_sparsity not in data[tensor_name]:
                    data[tensor_name][tensor_sparsity] = [None, None]

                if system == "TACO":
                    data[tensor_name][tensor_sparsity][0] = value
                elif system == "ATL":
                    data[tensor_name][tensor_sparsity][1] = value

                i += 2
            else:
                i += 1

    result = {}

    for tensor_name, sparsity_map in data.items():
        result[tensor_name] = {}
        for sparsity, vals in sparsity_map.items():
            taco, atl = vals

            if taco is None or atl is None:
                raise ValueError(
                    f"Incomplete data for tensor='{tensor_name}', sparsity='{sparsity}': {vals}"
                )

            if atl == 0:
                ratio = float("inf")  # or raise an error if you prefer
            else:
                ratio = taco / atl

            result[tensor_name][sparsity] = (taco, atl, ratio)

    return result

if __name__ == "__main__":
    directory = "./"  # change this
    result = process_files(directory)

    for tensor, sparsity_map in result.items():
        with open(f"{tensor}.csv","w") as F:
            print(f"Tensor: {tensor}")
            print("Sparsity, TACO, ATL, TACO/ATL")
            F.write("Sparsity, TACO, ATL, TACO/ATL\n")
            for sparsity, (t,a,r) in sparsity_map.items():
                print(f"{sparsity}, {t}, {a}, {r}")
                F.write(f"{sparsity}, {t}, {a}, {r}\n")
