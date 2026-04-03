import os
import glob
from collections import defaultdict

def process_files(directory):
    # format_name -> { matrix_name: [taco_val, atl_val] }
    data = defaultdict(dict)

    for filepath in glob.glob(os.path.join(directory, "*.mtx.txt")):
        filename = os.path.basename(filepath)
        matrix_name = filename.replace(".mtx.txt", "")

        with open(filepath, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        i = 0
        while i < len(lines):
            header = lines[i]
            value = float(lines[i + 1])

            parts = header.split()
            system = parts[0]         # "TACO" or "ATL"
            format_name = parts[1]   # assumes no spaces

            if matrix_name not in data[format_name]:
                data[format_name][matrix_name] = [None, None]

            if system == "Taco":
                data[format_name][matrix_name][0] = value
            elif system == "ATL":
                data[format_name][matrix_name][1] = value

            i += 2

    result = {}

    for format_name, matrix_map in data.items():
        result[format_name] = {}
        for matrix, vals in matrix_map.items():
            taco, atl = vals

            if taco is None or atl is None:
                raise ValueError(
                    f"Incomplete data for matrix='{matrix}', format='{format_name}': {vals}"
                )

            if atl == 0:
                ratio = float("inf")  # or raise an error if you prefer
            else:
                ratio = taco / atl

            result[format_name][matrix] = (taco, atl, ratio)

    return result

if __name__ == "__main__":
    directory = "./"  # change this
    result = process_files(directory)

    for fmt, matrix_map in result.items():
        print(f"\nFormat: {fmt}")
        for matrix, values in matrix_map.items():
            print(f"  {matrix}: {values}")
