
import argparse
import yaml
import sys
import pandas as pd
from pathlib import Path
from matnexus import VecGenerator


class MaterialProcessor:
    def __init__(self, config):
        self.config = config

        # Periodic table elements for verifying columns in material files
        self.PERIODIC_TABLE_ELEMENTS = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
            'Si', 'P',
            'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As',
            'Se',
            'Br', 'Kr', 'Rb', 'Sr', 'Y',
            'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
            'Te', 'I',
            'Xe', 'Cs', 'Ba', 'La', 'Ce',
            'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf',
            'Ta', 'W', 'Re', 'Os', 'Ir',
            'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac',
            'Th', 'Pa',
            'U', 'Np', 'Pu', 'Am', 'Cm',
            'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs',
            'Mt', 'Ds',
            'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
            'Lv', 'Ts', 'Og'
        ]

    def load_model(self, model_path):
        """
        Load a trained Word2Vec model from disk.

        :param model_path: Path to the saved Word2Vec model
        """
        # Use matnexus to load the model
        self.word2vec_model = VecGenerator.Word2VecModel.load(str(model_path))
        print(f"Model loaded from {model_path}")

    def process_materials(self, materials_dir, output_dir):
        """
        Process all material CSV files in 'materials_dir' using the loaded model.

        :param materials_dir: Directory containing material CSV files
        :param output_dir: Directory to store processed output files
        """
        output_dir = Path(output_dir)
        processed_data_dir = output_dir / "processed_data"
        processed_data_dir.mkdir(parents=True, exist_ok=True)

        params = self.config['materials_processing']
        property_list = params['property_list']

        # For each CSV file in the materials directory
        material_files = list(Path(materials_dir).glob("*.csv"))
        for material_file in material_files:
            print(f"Processing material file: {material_file.name}")
            material_df = pd.read_csv(material_file)

            # Create a similarity calculator for this model
            calculator = VecGenerator.MaterialSimilarityCalculator(self.word2vec_model)

            # Identify element columns
            elements = [col for col in material_df.columns if
                        col in self.PERIODIC_TABLE_ELEMENTS]

            # Calculate similarity for each specified property
            for prop in property_list:
                similarity_col = f"Similarity_to_{prop}"
                try:
                    temp_df = calculator.calculate_similarity_from_dataframe(
                        material_df,
                        elements,
                        target_property=[prop],
                        add_experimental_indicator=False
                    )
                    material_df[similarity_col] = temp_df['Similarity']
                except Exception as e:
                    print(f"Error calculating similarity for property '{prop}': {e}")
                    material_df[similarity_col] = pd.NA

            # Save the processed material file
            processed_file_name = f"{material_file.stem}_with_similarity.csv"
            processed_file_path = processed_data_dir / processed_file_name
            material_df.to_csv(processed_file_path, index=False)
            print(f"Processed file saved to {processed_file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Load a Word2Vec model and process materials.")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the YAML config file."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained Word2Vec model file."
    )
    parser.add_argument(
        "--materials_dir",
        type=str,
        required=True,
        help="Path to the directory containing material CSV files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for processed files."
    )
    args = parser.parse_args()

    # Read config from YAML file
    config_path = Path(args.config_file)
    if not config_path.is_file():
        print(f"Error: config file {config_path} not found.")
        sys.exit(1)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    processor = MaterialProcessor(config)
    processor.load_model(args.model_path)
    processor.process_materials(args.materials_dir, args.output_dir)


if __name__ == "__main__":
    main()