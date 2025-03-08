

import argparse
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def str2bool(v):
    """
    Convert a string to a boolean, for argparse convenience.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

class CentroidPlotter:
    def __init__(self, input_dir, output_dir):
        """
        Initialize the CentroidPlotter class.

        Parameters:
        - input_dir (str or Path): Path to the directory containing centroid history CSV files.
        - output_dir (str or Path): Path to save the output plots.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_centroid_data(self):
        """
        Load all centroid history files from the input directory and its subdirectories.

        Returns:
        - dict: A dictionary with filenames (stem) as keys and DataFrames as values.
        """
        centroid_data = {}
        for file in self.input_dir.rglob("*centroid_history.csv"):  # Search recursively
            df = pd.read_csv(file)
            centroid_data[file.stem] = df
        return centroid_data

    def _extract_material_system(self, file_stem):
        """
        Extract the material system name from the file stem.

        Parameters:
        - file_stem (str): The file name stem (without extension).

        Returns:
        - str: Material system name (everything before '_material_system' or '_centroid_history').
        """
        return file_stem.split("_material_system")[0].split("_centroid_history")[0].strip("_")

    def plot_iterations_vs_distance(self, save_individual=True, combined_plot=False,
                                    interval=5):
        """
        Plot iteration vs. distance for all centroid history files.

        Parameters:
        - save_individual (bool): Whether to save individual plots for each file.
        - combined_plot (bool): Whether to create a combined plot for all files.
        - interval (int): Interval for displaying x-axis ticks.
        """
        data = self.load_centroid_data()

        # Sort dictionary by extracted material system name
        sorted_data = {
            k: data[k] for k in sorted(data, key=self._extract_material_system)
        }

        # Individual plots
        if save_individual:
            for name, df in sorted_data.items():
                material_system = self._extract_material_system(name)
                plt.figure()
                plt.plot(df["step"], df["distance_from_previous"], marker="o",
                         label=material_system)
                plt.xlabel("Iteration")
                plt.ylabel("Centroid Distance")
                plt.legend(title=None)

                # Adjust x-axis ticks
                step_min, step_max = df["step"].min(), df["step"].max()
                plt.xticks(range(step_min, step_max + 1, interval))

                output_file = self.output_dir / f"{name}_iteration_vs_distance.pdf"
                plt.savefig(output_file, bbox_inches="tight")
                plt.close()
                print(f"Saved individual plot for {material_system}: {output_file}")

        # Combined plot
        if combined_plot:
            plt.figure()
            for name, df in sorted_data.items():
                material_system = self._extract_material_system(name)
                plt.plot(df["step"], df["distance_from_previous"], marker="o",
                         label=material_system.replace("_", ""))
            plt.xlabel("Iteration")
            plt.ylabel("Centroid Distance")
            plt.legend(title=None)

            # Adjust x-axis ticks
            all_steps = sorted(
                set(step for df in sorted_data.values() for step in df["step"]))
            step_min, step_max = min(all_steps), max(all_steps)
            plt.xticks(range(step_min, step_max + 1, interval))

            output_file = self.output_dir / "combined_iteration_vs_distance.pdf"
            plt.savefig(output_file, bbox_inches="tight")
            plt.close()
            print(f"Saved combined plot: {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot centroid histories from CSV files."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=str,
        help="Path to the input directory containing centroid_history CSV files."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Path to the output directory to save the plots."
    )
    parser.add_argument(
        "--save_individual",
        type=str2bool,
        default="true",
        help="Whether to save individual plots (true/false). Default='true'."
    )
    parser.add_argument(
        "--combined_plot",
        type=str2bool,
        default="false",
        help="Whether to create one combined plot of all files (true/false). Default='false'."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    plotter = CentroidPlotter(args.input_dir, args.output_dir)
    plotter.plot_iterations_vs_distance(
        save_individual=args.save_individual,
        combined_plot=args.combined_plot
    )

if __name__ == "__main__":
    main()