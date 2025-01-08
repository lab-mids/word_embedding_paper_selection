import argparse
import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import json


class MultiObjectiveParetoAnalyzer:
    def __init__(self, objectives, global_direction=None, specific_directions=None):
        self.objectives = objectives
        self.global_direction = global_direction if global_direction else ['max'] * len(
            objectives)
        self.specific_directions = specific_directions if specific_directions else {}

    def calculate_pareto_front(self, dataframe, df_key):
        scores = dataframe[self.objectives].to_numpy()
        population_size = scores.shape[0]
        pareto_front = np.ones(population_size, dtype=bool)

        for I in range(population_size):
            for j in range(I + 1, population_size):
                if self.is_dominated(scores[I], scores[j], df_key):
                    pareto_front[I] = False
                    break
                elif self.is_dominated(scores[j], scores[I], df_key):
                    pareto_front[j] = False

        return dataframe[pareto_front]

    def is_dominated(self, x, y, df_key):
        directions = self.specific_directions.get(df_key, self.global_direction)
        for I, direction in enumerate(directions):
            if direction == 'min':
                if not (y[I] <= x[I]) or (x[I] < y[I]):
                    return False
            elif direction == 'max':
                if not (y[I] >= x[I]) or (x[I] > y[I]):
                    return False
        return True

    def process_file(self, file_path, output_directory):
        df_key = os.path.basename(file_path).rsplit('.', 1)[0]
        dataframe = pd.read_csv(file_path)
        pareto_front_df = self.calculate_pareto_front(dataframe, df_key)

        output_filename = f'{df_key}_pareto_front.csv'
        output_path = os.path.join(output_directory, output_filename)

        pareto_front_df.to_csv(output_path, index=False)


def process_all_files_in_directory(input_directory, output_directory, objectives,
                                   global_direction=None, specific_directions=None,
                                   num_workers=4):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    filenames = [f for f in os.listdir(input_directory) if
                 f.endswith('_material_system_with_similarity.csv')]

    analyzer = MultiObjectiveParetoAnalyzer(objectives, global_direction,
                                            specific_directions)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(analyzer.process_file,
                            os.path.join(input_directory, filename), output_directory)
            for filename in filenames
        ]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f'Error processing file: {e}')


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Pareto front for datasets")
    parser.add_argument("--input_directory", type=str, required=True,
                        help="Input directory path")
    parser.add_argument("--output_directory", type=str, required=True,
                        help="Output directory path")
    parser.add_argument("--objectives", type=str, required=True,
                        help="List of objectives")
    parser.add_argument("--global_direction", type=str, required=True,
                        help="Global direction list")
    parser.add_argument("--specific_directions", type=str, required=True,
                        help="Specific directions dictionary")
    parser.add_argument("--num_workers", type=int, required=True,
                        help="Number of workers")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    objectives = json.loads(args.objectives)
    global_direction = json.loads(args.global_direction)
    specific_directions = json.loads(args.specific_directions)

    process_all_files_in_directory(args.input_directory, args.output_directory,
                                   objectives, global_direction, specific_directions,
                                   args.num_workers)


if __name__ == "__main__":
    main()