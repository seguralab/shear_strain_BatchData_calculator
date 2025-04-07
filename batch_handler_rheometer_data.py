#README

#This code expects a data folder with CSV files containing rheometer data.
#Please see the sample CSV attached to the repository for the expected format.
#It processes each file, calculates the storage and shear moduli for the first THREE cycles,
#and saves the results in a summary CSV file.
#The code uses pandas for data manipulation and numpy for numerical calculations.
#It assumes that the CSV files have specific columns for ZTip Displacement, ZForce, and Current Size.
#The code also handles the case where the CSV files may not have the expected format or data.
#The code is designed to be run in a Python environment with the necessary libraries installed.
#This should be run in a Python 3.x environment with pandas and numpy installed.

#This code should be able to run in any compiler (mine was Visual Studio Code) as long as the required libraries and python is installed.



import os
import glob
import pandas as pd
import numpy as np

def calculate_moduli_by_cycle(df):
    """
    Add a new column 'Cycle_Number' that increments every time a new compress phase starts.
    Then, filter the compress rows (rows where the 'Cycle' value ends with 'Compress')
    and compute moduli for each cycle using the loading phase.
    """
    df = df.copy()
    cycle_numbers = []
    current_cycle = 0
    # Iterate row‐by‐row; when a row’s Cycle ends with "Compress" and it's not the very first row,
    # assume it starts a new cycle (if the previous row was not a Compress row).
    for idx, row in df.iterrows():
        cycle_val = row["Cycle"].strip()
        if cycle_val.endswith("Compress"):
            if idx == 0:
                current_cycle = 1
            else:
                prev_val = df.loc[idx - 1, "Cycle"].strip()
                # When a new "Compress" appears after a Hold or Recover, start a new cycle.
                if not prev_val.endswith("Compress"):
                    current_cycle += 1
            cycle_numbers.append(current_cycle)
        else:
            cycle_numbers.append(current_cycle)
    df["Cycle_Number"] = cycle_numbers

    # Filter only the compress-phase rows
    compress_df = df[df["Cycle"].str.strip().str.endswith("Compress")].copy()
    results = {}
    # Compute moduli for each cycle group
    for cycle, group in compress_df.groupby("Cycle_Number"):
        # For each group, set adjusted displacement and stress from the file columns
        group["Adjusted Displacement (μm)"] = group["ZTip Displacement(um)"]
        group["Stress (μPa)"] = group["ZForce(uN)"]
        # Compute strain using the first row as baseline
        initial_disp = group["Adjusted Displacement (μm)"].iloc[0]
        initial_length = group["Current Size (um)"].iloc[0]
        group["Strain"] = (group["Adjusted Displacement (μm)"] - initial_disp) / initial_length

        # Select the loading phase (where displacement is increasing)
        d_disp = np.gradient(group["Adjusted Displacement (μm)"])
        loading_group = group[d_disp > 0]
        if len(loading_group) < 2:
            continue  # Not enough data for this cycle
        strain = loading_group["Strain"]
        stress = loading_group["Stress (μPa)"]
        coeffs = np.polyfit(strain, stress, 1)
        E_prime = coeffs[0]      # Storage modulus (slope)
        G_prime = E_prime / 3    # Shear modulus
        results[cycle] = {"Storage Modulus (μPa)": E_prime,
                          "Shear Modulus (μPa)": G_prime}
    return results

def process_file(file_path):
    """
    Reads a CSV file (using latin1 encoding) and calculates moduli for each cycle.
    Returns the file name and a dictionary of cycle moduli.
    """
    try:
        df = pd.read_csv(file_path, encoding='latin1')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None
    moduli_dict = calculate_moduli_by_cycle(df)
    test_name = os.path.basename(file_path)
    return test_name, moduli_dict

def main():
    # Folder containing the CSV files; adjust the path if necessary.
    folder = "data"
    file_list = glob.glob(os.path.join(folder, "*.csv"))
    output_rows = []
    for file in sorted(file_list):
        test_name, moduli_dict = process_file(file)
        if moduli_dict is None or len(moduli_dict) < 3:
            print(f"Skipping file {test_name} due to insufficient cycles (found {len(moduli_dict)}).")
            continue
        # Use the cycle numbers in ascending order and take the first 3 cycles
        cycles = sorted(moduli_dict.keys())
        cycle1 = moduli_dict[cycles[0]]
        cycle2 = moduli_dict[cycles[1]]
        cycle3 = moduli_dict[cycles[2]]
        row = {
            "File Name": test_name,
            "Cycle 1 Storage Modulus (μPa)": cycle1["Storage Modulus (μPa)"],
            "Cycle 1 Shear Modulus (μPa)": cycle1["Shear Modulus (μPa)"],
            "Cycle 2 Storage Modulus (μPa)": cycle2["Storage Modulus (μPa)"],
            "Cycle 2 Shear Modulus (μPa)": cycle2["Shear Modulus (μPa)"],
            "Cycle 3 Storage Modulus (μPa)": cycle3["Storage Modulus (μPa)"],
            "Cycle 3 Shear Modulus (μPa)": cycle3["Shear Modulus (μPa)"]
        }
        output_rows.append(row)
    if output_rows:
        output_df = pd.DataFrame(output_rows)
        output_csv = "moduli_summary.csv"
        output_df.to_csv(output_csv, index=False)
        print(f"Summary of moduli saved to {output_csv}")
    else:
        print("No files processed.")

if __name__ == "__main__":
    main()
