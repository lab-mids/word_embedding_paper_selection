

import argparse
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


class CurrentDensityPlot:
    def __init__(self, combined_csv_path, material_systems, sort_by="name"):
        """
        Initialize the class with the combined dataset and material system names.

        Args:
            combined_csv_path (str): Path to the combined CSV file containing all material system comparisons.
            material_systems (list): List of material system names to include in the plot.
            sort_by (str): Sorting criteria for material systems. Options: "name", "custom".
        """
        # Load the combined data
        self.data = pd.read_csv(combined_csv_path)

        # Filter for the specified material systems
        self.data = self.data[self.data["Material_System"].isin(material_systems)]

        # Sort material systems
        if sort_by == "name":
            self.data.sort_values("Material_System", inplace=True)
        elif sort_by == "custom":
            # Sort by length of material system name, then alphabetically
            self.data.sort_values(
                by=["Material_System"],
                key=lambda col: col.str.len(),
                inplace=True
            )

        # Extract the unique material systems and metric label
        self.material_systems = sorted(self.data["Material_System"].unique().tolist())
        self.metric_label = self.data["Metric"].iloc[0]  # Assume all rows have the same metric

    def plot(self, show="Min", figsize=(8, 6), text_size=12):
        """
        Create a bar plot for Current Density Stats by Metrics and return the figure.

        Args:
            show (str): Either "Min" or "Max" to determine which values to display.
            figsize (tuple): The figure size for the plot.
            text_size (int): The font size for all text elements in the plot.

        Returns:
            fig: The matplotlib figure object for further use or saving.
        """
        # Validate `show` input
        if show not in ["Min", "Max"]:
            raise ValueError("`show` must be either 'Min' or 'Max'.")

        # Prepare data for plotting
        values = []
        # Group the data by material system
        for _, group in self.data.groupby("Material_System"):
            if show == "Min":
                values.append([
                    group["Ori_Min"].iloc[0],
                    group["Full_Min"].iloc[0],
                    group["Selection_Min"].iloc[0]
                ])
            else:  # show == "Max"
                values.append([
                    group["Ori_Max"].iloc[0],
                    group["Full_Max"].iloc[0],
                    group["Selection_Max"].iloc[0]
                ])

        # Convert to DataFrame for plotting
        bar_data = pd.DataFrame(values, columns=["Original", "Full", "Selection"])
        bar_data["Material_System"] = self.material_systems

        # Melt data for Seaborn
        bar_data_melted = bar_data.melt(
            id_vars="Material_System",
            var_name="Type",
            value_name="Value"
        )

        # Create the Seaborn bar plot
        sns.set_theme(style="whitegrid")
        palette = sns.color_palette("Set2", n_colors=3)

        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(
            data=bar_data_melted,
            x="Material_System",
            y="Value",
            hue="Type",
            palette=palette,
            dodge=True,
            ax=ax
        )

        # Add hatching styles for each bar group
        hatches = {
            "Original": None,  # No hatching for Original
            "Full": "//",      # Diagonal lines for Full
            "Selection": "xx"  # Cross hatching for Selection
        }

        # Apply hatch style to each bar
        for patch, (_, row) in zip(ax.patches, bar_data_melted.iterrows()):
            patch.set_hatch(hatches[row["Type"]])

        # Adjust bar width if only one material system
        if len(self.material_systems) == 1:
            for bar in ax.patches:
                bar.set_width(0.1)  # Narrower bars for single-system case

        # Annotate bars
        for bar in ax.patches:
            height = bar.get_height()
            if height != 0:  # Avoid labeling zero-height bars
                position = height + 0.01 if height > 0 else height - 0.01
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    position,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom" if height > 0 else "top",
                    fontsize=text_size - 2,
                    color="black"
                )

        # Customize the legend to reflect hatching patterns and correct colors
        custom_handles = [
            Patch(facecolor=palette[0], hatch=None, label="Original"),
            Patch(facecolor=palette[1], hatch="//", label="Full"),
            Patch(facecolor=palette[2], hatch="xx", label="Selection")
        ]
        ax.legend(handles=custom_handles, fontsize=text_size - 2)

        # Customize the plot
        ylabel = f"{show} value of {self.metric_label} (mA/cm²)"
        ax.set_xlabel("Material System", fontsize=text_size)
        ax.set_ylabel(ylabel, fontsize=text_size)
        ax.set_xticks(range(len(self.material_systems)))
        ax.set_xticklabels(self.material_systems, fontsize=text_size)
        fig.tight_layout()

        return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a Current Density comparison plot."
    )
    parser.add_argument(
        "--combined_csv_path",
        type=str,
        required=True,
        help="Path to the combined CSV file."
    )
    parser.add_argument(
        "--material_systems",
        type=str,
        required=True,
        help="JSON list of material system names, e.g. '[\"Ag_Pd_Pt\",\"Ag_Pd_Pt_Ru\"]'"
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default="name",
        choices=["name", "custom"],
        help="How to sort material systems. Default='name'."
    )
    parser.add_argument(
        "--show",
        type=str,
        default="Min",
        choices=["Min", "Max"],
        help="Whether to show 'Min' or 'Max' values in the bar chart."
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="[10, 8]",
        help="Figure size as JSON list, e.g. '[10, 8]'."
    )
    parser.add_argument(
        "--text_size",
        type=int,
        default=14,
        help="Font size for labels and annotations."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the resulting plot file (e.g. a PDF)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse JSON strings where relevant
    material_systems = json.loads(args.material_systems)
    figsize = json.loads(args.figsize)

    plotter = CurrentDensityPlot(
        combined_csv_path=args.combined_csv_path,
        material_systems=material_systems,
        sort_by=args.sort_by
    )

    fig = plotter.plot(show=args.show, figsize=tuple(figsize), text_size=args.text_size)

    # Save figure
    fig.savefig(args.output_file, bbox_inches="tight")
    print(f"Plot saved to: {args.output_file}")


if __name__ == "__main__":
    main()