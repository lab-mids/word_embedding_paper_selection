import os
import sys
import argparse
import pandas as pd
from pathlib import Path


def parse_args():
    """
    Parse command line arguments to allow specifying paths as parameters.
    """
    parser = argparse.ArgumentParser(description="Compare CSV stats for material systems.")
    parser.add_argument("--full_results_dir", type=str, required=True, help="Path to the full results directory.")
    parser.add_argument("--selection_results_dir", type=str, required=True, help="Path to the selection results directory.")
    parser.add_argument("--ori_dir", type=str, required=False, help="Path to the original directory.")
    parser.add_argument("--paper_selection_dir", type=str, required=True, help="Path to the paper selection directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    return parser.parse_args()


class MaterialStatsComparer:
    def __init__(self, full_results_dir, selection_results_dir, ori_dir, paper_selection_dir, output_dir):
        """
        Initialize the comparer with directories for full results, selection results,
        original data, paper selection, and output.
        """
        self.full_results_dir = Path(full_results_dir)
        self.selection_results_dir = Path(selection_results_dir)
        self.ori_dir = Path(ori_dir) if ori_dir else None
        self.paper_selection_dir = Path(paper_selection_dir)
        self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compare_stats(self):
        """
        Compare statistics of paired files and save the concatenated results.
        """
        full_files = {f.name: f for f in self.full_results_dir.rglob("*.csv")}
        selection_files = {f.name: f for f in self.selection_results_dir.rglob("*.csv")}
        ori_files = {f.name: f for f in self.ori_dir.rglob("*.csv")} if self.ori_dir else {}
        paper_selection_files = {f.name: f for f in self.paper_selection_dir.rglob("*.csv")}

        all_stats = []

        for filename in full_files:
            if filename in selection_files:
                material_system = "_".join(filename.split('_')[:-6])
                full_file_path = full_files[filename]
                selection_file_path = selection_files[filename]

                ori_filename = f"{material_system}_material_system.csv"
                paper_selection_filename = f"{material_system}_material_system_selected_papers.csv"
                ori_file_path = ori_files.get(ori_filename, None)
                paper_selection_file_path = paper_selection_files.get(paper_selection_filename, None)

                full_df = pd.read_csv(full_file_path)
                selection_df = pd.read_csv(selection_file_path)
                ori_df = pd.read_csv(ori_file_path) if ori_file_path else None
                paper_selection_df = pd.read_csv(paper_selection_file_path) if paper_selection_file_path else None

                stats = self._compare_file_stats(full_df, selection_df, ori_df, paper_selection_df)
                stats['Material_System'] = material_system

                all_stats.append(stats)

        if all_stats:
            combined_stats = pd.concat(all_stats, ignore_index=True)
        else:
            print("No matching CSV files found to compare.")
            sys.exit(0)

        output_file = self.output_dir / "combined_comparison_results.csv"
        combined_stats.to_csv(output_file, index=False)
        print(f"Combined comparison results saved to: {output_file}")

    def _compare_file_stats(self, full_df, selection_df, ori_df, paper_selection_df):
        current_cols = [col for col in full_df.columns if col.startswith("Current")]
        if not current_cols:
            raise ValueError("No column starting with 'Current' found in the dataset.")

        stats_list = []
        for col in current_cols:
            full_stats = full_df[col].describe()
            selection_stats = selection_df[col].describe()
            ori_stats = ori_df[col].describe() if ori_df is not None and col in ori_df.columns else None

            stats = {
                "Metric": col,
                "Ori_Entries": len(ori_df) if ori_df is not None else 0,
                "Full_Entries": len(full_df),
                "Selection_Entries": len(selection_df),
                "Ori_Min": ori_stats["min"] if ori_stats is not None else None,
                "Full_Min": full_stats["min"],
                "Selection_Min": selection_stats["min"],
                "Ori_Max": ori_stats["max"] if ori_stats is not None else None,
                "Full_Max": full_stats["max"],
                "Selection_Max": selection_stats["max"],
                "Selected_Papers": len(paper_selection_df) if paper_selection_df is not None else 0,
            }

            stats_list.append(stats)

        return pd.DataFrame(stats_list)


def main():
    """
    Main entry point for running the material stats comparison.
    """
    args = parse_args()

    comparer = MaterialStatsComparer(
        args.full_results_dir,
        args.selection_results_dir,
        args.ori_dir,
        args.paper_selection_dir,
        args.output_dir
    )
    comparer.compare_stats()


if __name__ == "__main__":
    main()