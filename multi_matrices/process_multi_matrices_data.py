import os
import glob
from collections import defaultdict

def process_files(directory):
    # kernel -> { tensor_name: [taco_val, atl_val, taco_val/atl_val] }
    data = defaultdict(dict)

    for filepath in glob.glob(os.path.join(directory, "*.txt")):
        filename = os.path.basename(filepath)
        matrix_name = filename.replace(".txt", "")
        
        with open(filepath, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        i = 0
        while i < len(lines):
            header = lines[i]
            parts = header.split()
            system = parts[0]
            if (system == "SpMSpV" or system == "ATL"):
                matrix_sparsity = parts[-1]
                
                value_line = lines[i+1]
                value_parts = value_line.split()
                if (1 < len(value_parts)):
                    i = i+1
                    value_line = lines[i+1]
                value = float(value_line)
                
                if matrix_sparsity not in data[matrix_name]:
                    data[matrix_name][matrix_sparsity] = [None, None]

                if system == "SpMSpV":
                    data[matrix_name][matrix_sparsity][0] = value
                elif system == "ATL":
                    data[matrix_name][matrix_sparsity][1] = value

                i += 2
            else:
                i += 1

    result = {}

    for matrix_name, sparsity_map in data.items():
        result[matrix_name] = {}
        for sparsity, vals in sparsity_map.items():
            taco, atl = vals

            if taco is None or atl is None:
                raise ValueError(
                    f"Incomplete data for matrix='{matrix_name}', sparsity='{ssparsity}': {vals}"
                )

            if atl == 0:
                ratio = float("inf")  # or raise an error if you prefer
            else:
                ratio = taco / atl

            result[matrix_name][sparsity] = (taco, atl, ratio)

    return result

if __name__ == "__main__":
    directory = "./"  # change this
    result = process_files(directory)

    for fmt, tensor_map in result.items():
        print(f"Matrix: {fmt}")
        for tensor, values in tensor_map.items():
            print(f"  {tensor}: {values}")
