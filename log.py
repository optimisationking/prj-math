import csv
import numpy as np
import problem as p

# columns written for every method
HEADER = ["k", "r", "h", "lambda", "L", "grad_norm", "error"]


def make_row(k, Xk, grad_norm, X_star, extra_values=()):
    # Return the data of iteration k as a list.
    row = [k, Xk[0], Xk[1], Xk[2], p.lagrangien(Xk), grad_norm,
           np.linalg.norm(Xk - X_star)]
    return row + list(extra_values)


def write_log(filename, rows, extra_columns=()):
    # Write the header and all the rows in a CSV file.
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER + list(extra_columns))
        writer.writerows(rows)