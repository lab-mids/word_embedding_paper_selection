

import argparse
import yaml
import sys
import pandas as pd
from pathlib import Path
from gensim.models import Word2Vec
from matnexus import VecGenerator


class SimpleWorkflowPipeline:
    def __init__(self, config):
        """
        Initialize the pipeline with a configuration dictionary.
        """
        self.config = config

        # Periodic table elements for checking columns in material files.
        self.PERIODIC_TABLE_ELEMENTS = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P',
            'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
            'Br', 'Kr', 'Rb', 'Sr', 'Y',
            'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
            'Xe', 'Cs', 'Ba', 'La', 'Ce',
            'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
            'Ta', 'W', 'Re', 'Os', 'Ir',
            'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
            'U', 'Np', 'Pu', 'Am', 'Cm',
            'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
            'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
            'Lv', 'Ts', 'Og'
        ]

    def train_word2vec(self, abstracts_df, output_path):
        """
        Train a Word2Vec model on the abstracts and save it.

        :param abstracts_df: DataFrame containing abstracts.
        :param output_path: Path to save the trained Word2Vec model.
        """
        params = self.config['word2vec']

        # Convert DataFrame into sentences via matnexus
        corpus = VecGenerator.Corpus(abstracts_df)
        sentences = corpus.sentences

        # Create and fit the model
        model = VecGenerator.Word2VecModel(sentences)
        model.fit(
            sg=params.get('sg', 1),
            vector_size=params.get('vector_size', 100),
            hs=params.get('hs', 0),
            window=params.get('window', 5),
            min_count=params.get('min_count', 1),
            workers=params.get('workers', 4)
        )

        # Save model
        model.save(str(output_path))
        self.word2vec_model = model

    def process_materials(self, material_df):
        """
        Process materials using the trained Word2Vec model to calculate similarity.

        :param material_df: DataFrame containing material data.
        :return: Updated DataFrame with similarity columns added.
        """
        params = self.config['materials_processing']
        property_list = params['property_list']
        elements = [col for col in material_df.columns if col in self.PERIODIC_TABLE_ELEMENTS]

        if not hasattr(self, 'word2vec_model'):
            raise ValueError("Word2Vec model not trained. Train the model before processing materials.")

        # Create similarity calculator
        self.calculator = VecGenerator.MaterialSimilarityCalculator(self.word2vec_model)

        for prop in property_list:
            similarity_col = f"Similarity_to_{prop}"
            try:
                temp_df = self.calculator.calculate_similarity_from_dataframe(
                    material_df,
                    elements,
                    target_property=[prop],
                    add_experimental_indicator=False
                )
                material_df[similarity_col] = temp_df['Similarity']
            except Exception as e:
                print(f"Error calculating similarity for property '{prop}': {e}")
                material_df[similarity_col] = pd.NA

        return material_df

    def run_workflow(self, abstracts_csv, materials_dir, output_dir):
        """
        Main workflow function.

        :param abstracts_csv: Path to the abstracts CSV file.
        :param materials_dir: Path to the directory containing material CSV files.
        :param output_dir: Path to the output directory for processed files and models.
        """
        output_dir = Path(output_dir)
        processed_data_dir = output_dir / "processed_data"
        output_models_dir = output_dir / "output_models"
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        output_models_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load abstracts and train Word2Vec
        abstracts_df = pd.read_csv(abstracts_csv)
        final_model_path = output_models_dir / "final_word2vec.model"
        self.train_word2vec(abstracts_df, final_model_path)

        # 2. Process each material file
        material_files = list(Path(materials_dir).glob("*.csv"))
        for material_file in material_files:
            print(f"Processing material file: {material_file.name}")
            material_df = pd.read_csv(material_file)
            processed_material_df = self.process_materials(material_df)

            # Save the processed material file
            processed_file_name = f"{material_file.stem}_with_similarity.csv"
            processed_file_path = processed_data_dir / processed_file_name
            processed_material_df.to_csv(processed_file_path, index=False)

        print("Workflow complete. Processed files saved.")


def main():
    parser = argparse.ArgumentParser(description="Run the SimpleWorkflowPipeline.")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the YAML config file."
    )
    parser.add_argument(
        "--abstracts_csv",
        type=str,
        required=True,
        help="Path to the abstracts CSV file."
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
        help="Path to the output directory for processed files and models."
    )

    args = parser.parse_args()

    # Read config from YAML file
    config_path = Path(args.config_file)
    if not config_path.is_file():
        print(f"Error: config file {config_path} not found.")
        sys.exit(1)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize and run the pipeline
    pipeline = SimpleWorkflowPipeline(config)
    pipeline.run_workflow(
        abstracts_csv=args.abstracts_csv,
        materials_dir=args.materials_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()