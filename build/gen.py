import numpy as np 
import random 

def generate_random_mtx(rows, cols, sparsity=0.5,filename="random_mat.mtx"):
    """
    Generates a random sparse matrix in MTX format and saves it to a file. 
    Args:
    rows (int): Number of rows in the matrix.
    cols (int): Number of columns in the matrix.
    filename (str): Output file name.
    """
    data = np.random.rand(rows * cols) 
    # Write to MTX file
    indices = random.sample(range(rows*cols),int((1-sparsity)*rows*cols))
    with open(filename,"w") as f:
        f.write("%%MatrixMarket matrix array real general\n")
        f.write(f"{rows} {cols}\n")
        for i in range(rows*cols):
            if i in indices:
                print(i)
                f.write(f"{int(data[i] * 10)}\n") 
            else:
                f.write("0\n") 
    
    print(f"Random MTX file generated: {filename}")
    

generate_random_mtx(200,200,0.80) 
