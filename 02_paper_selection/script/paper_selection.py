import argparse
import sys

import pandas as pd
import yaml
from sklearn.decomposition import PCA
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from matnexus import VecGenerator
import os
from pathlib import Path



class WorkflowPipeline:
    def __init__(self, config):
        """
        Initialize the pipeline with a configuration dictionary.
        """
        self.config = config
        self.random_seed = config.get('random_seed', 42)
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
        np.random.seed(self.random_seed)

    def train_doc2vec(self, abstracts_df):
        params = self.config['doc2vec']

        # Filter and create a new copy to avoid SettingWithCopyWarning
        abstracts_df = abstracts_df[abstracts_df['abstract'].notna()].copy()
        abstracts_df['abstract'] = abstracts_df['abstract'].astype(str)

        tagged_docs = [
            TaggedDocument(words=abstract.split(), tags=[str(i)])
            for i, abstract in enumerate(abstracts_df['abstract'])
        ]
        model = Doc2Vec(
            tagged_docs,
            vector_size=params.get('vector_size', 50),
            window=params.get('window', 5),
            min_count=params.get('min_count', 1),
            workers=params.get('workers', 4),
            seed=self.random_seed
        )
        self.doc2vec_model = model
        self.paper_vectors = np.array([model.dv[str(i)] for i in range(len(tagged_docs))])
        return model

    def train_word2vec(self, abstracts_df, output_path=None):
        params = self.config['word2vec']
        corpus = VecGenerator.Corpus(abstracts_df)
        sentences = corpus.sentences
        model = VecGenerator.Word2VecModel(sentences)
        model.fit(
            sg=params.get('sg', 1),
            vector_size=params.get('vector_size', 100),
            hs=params.get('hs', 1),
            window=params.get('window', 5),
            min_count=params.get('min_count', 1),
            workers=params.get('workers', 4)
        )
        if output_path:  # Save only if an output path is specified
            model.save(str(output_path))  # Convert Path to string
        self.word2vec_model = model  # Keep the model in memory
        return model

    def process_materials(self, material_df):
        params = self.config['materials_processing']
        property_list = params['property_list']
        elements = [col for col in material_df.columns if col in self.PERIODIC_TABLE_ELEMENTS]

        if not hasattr(self, 'word2vec_model'):
            raise ValueError("Word2Vec model not found. Ensure the model is trained before processing materials.")

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

                if 'Material_Vec' in temp_df.columns:
                    material_df['Material_Vec'] = temp_df['Material_Vec']
                else:
                    material_df['Material_Vec'] = np.nan
            except Exception as e:
                material_df[similarity_col] = np.nan

        return material_df
    def greedy_selection(self):
        """
        Optimized greedy selection:
        - Uses vectorized operations for distance calculations.
        - Tracks minimum distances for unselected vectors to reduce redundant computations.
        """
        params = self.config['greedy_selection']
        vectors = self.paper_vectors

        if params.get('use_pca', False):
            n_components = params.get('n_components', 2)
            pca = PCA(n_components=n_components)
            vectors = pca.fit_transform(vectors)

        # Parameters
        start_size = params.get('start_size', 50)
        step_size = params.get('step_size', 50)
        method = params.get('method', 'cosine')

        # Initialize selection for the first step
        if not hasattr(self, 'selected_indices'):
            # Compute the center of the vector space
            vector_center = vectors.mean(axis=0)

            # Calculate distances to the center
            if method == 'cosine':
                from scipy.spatial.distance import cdist
                distances_to_center = cdist(vectors, vector_center.reshape(1, -1), metric='cosine').flatten()
            elif method == 'euclidean':
                distances_to_center = np.linalg.norm(vectors - vector_center, axis=1)
            else:
                raise ValueError("Unsupported distance method. Choose 'cosine' or 'euclidean'.")

            # Find the closest vector to the center
            first_index = np.argmin(distances_to_center)

            # Initialize the selected and remaining indices
            self.selected_indices = [first_index]
            self.remaining_indices = list(range(len(vectors)))
            self.remaining_indices.remove(first_index)

            # Calculate initial distances from all remaining vectors to the first selected vector
            distances = self._compute_distances(vectors, first_index, method)
            self.min_distances = distances.copy()  # Track minimum distances for efficiency

            # Select the rest of the first batch
            while len(self.selected_indices) < start_size:
                self._select_next_furthest(vectors, method)

            # Set the target size for subsequent steps
            self.target_size = start_size

        else:
            # Increment target size for this batch
            self.target_size += step_size

            # Add only the new `step_size` papers
            while len(self.selected_indices) < self.target_size:
                self._select_next_furthest(vectors, method)

        return self.selected_indices

    def _select_next_furthest(self, vectors, method):
        """
        Helper method to select the next furthest vector using tracked minimum distances.
        """
        # Identify the furthest vector from the selected set
        furthest_index = np.argmax(self.min_distances)
        actual_index = self.remaining_indices[furthest_index]

        # Append the furthest vector to the selected set
        self.selected_indices.append(actual_index)

        # Remove the selected vector from the remaining set
        del self.remaining_indices[furthest_index]

        # Remove the corresponding entry in min_distances
        self.min_distances = np.delete(self.min_distances, furthest_index)

        # Compute new distances for the remaining vectors
        new_distances = self._compute_distances(vectors, actual_index, method)

        # Update the minimum distances
        self.min_distances = np.minimum(self.min_distances, new_distances)

    def _compute_distances(self, vectors, selected_index, method):
        """
        Compute distances from one vector to all remaining vectors using the specified method.
        """
        remaining_vectors = vectors[self.remaining_indices]
        selected_vector = vectors[selected_index].reshape(1, -1)

        if method == 'cosine':
            from scipy.spatial.distance import cdist
            distances = cdist(remaining_vectors, selected_vector, metric='cosine').flatten()
        elif method == 'euclidean':
            distances = np.linalg.norm(remaining_vectors - selected_vector, axis=1)
        else:
            raise ValueError("Unsupported distance method. Choose 'cosine' or 'euclidean'.")

        return distances

    def calculate_centroid(self, material_df, similarity_cols):
        """
        Calculate the centroid based on the similarity columns.

        Parameters:
        - material_df (pd.DataFrame): The DataFrame containing material similarities.
        - similarity_cols (list): The list of columns representing similarity scores.

        Returns:
        - np.array: The centroid of the similarity scores, or None if any values are missing.
        """
        if material_df[similarity_cols].isnull().any().any():
            return None
        similarity_data = material_df[similarity_cols].values
        return similarity_data.mean(axis=0)

    def _reset_selection_state(self):
        if hasattr(self, 'selected_indices'):
            del self.selected_indices
        if hasattr(self, 'remaining_indices'):
            del self.remaining_indices
        if hasattr(self, 'min_distances'):
            del self.min_distances
        if hasattr(self, 'target_size'):
            del self.target_size

    def run_workflow(self, abstracts_csv, materials_dir, output_dir):
        """
        Main workflow function for processing multiple material files.

        Parameters:
        - abstracts_csv: Path to the abstracts CSV file.
        - materials_dir: Path to the directory containing material CSV files.
        - output_dir: Path to the output directory for processed files and models.
        """
        self._reset_selection_state()

        # Ensure the main output directory and subdirectories exist
        output_dir = Path(output_dir)
        processed_data_dir = output_dir / "processed_data"
        output_models_dir = output_dir / "output_models"
        selected_papers_dir = output_dir / "selected_papers"
        centroid_history_dir = output_dir / "centroid_history"
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        output_models_dir.mkdir(parents=True, exist_ok=True)
        selected_papers_dir.mkdir(parents=True, exist_ok=True)
        centroid_history_dir.mkdir(parents=True, exist_ok=True)

        # Load and process abstracts
        abstracts_df = pd.read_csv(abstracts_csv)
        self.train_doc2vec(abstracts_df)

        # Iterate through material files in the directory
        material_files = list(Path(materials_dir).glob("*.csv"))
        for material_file in material_files:
            print(f"Processing material file: {material_file.name}")

            # Initialize variables for greedy selection
            prev_centroid = None
            valid_centroid = False
            step = 1
            accumulated_selected_indices = set()
            centroid_history = []
            self._reset_selection_state()

            while True:
                selected_indices = self.greedy_selection()
                accumulated_selected_indices.update(selected_indices)

                # Get the selected papers
                selected_papers = abstracts_df.iloc[list(accumulated_selected_indices)]

                # Save the selected papers for the current material file
                selected_papers_name = f"{material_file.stem}_selected_papers.csv"
                selected_papers_path = selected_papers_dir / selected_papers_name
                selected_papers.to_csv(selected_papers_path, index=False)

                # Train Word2Vec model for the selected papers
                self.train_word2vec(selected_papers, None)

                # Process the current material file
                material_df = pd.read_csv(material_file)
                processed_material_df = self.process_materials(material_df)

                # Save the processed material file
                processed_file_name = f"{material_file.stem}_with_similarity.csv"
                processed_file_path = processed_data_dir / processed_file_name
                processed_material_df.to_csv(processed_file_path, index=False)

                # Calculate the centroid and check stopping condition
                similarity_cols = self.config['materials_processing']['similarity_cols']
                centroid = self.calculate_centroid(processed_material_df, similarity_cols)

                if centroid is None:
                    step += 1
                    continue

                if valid_centroid and prev_centroid is not None:
                    distance = np.linalg.norm(centroid - prev_centroid)
                    print(f"Distance between centroids (step {step - 1} and {step}): {distance}")

                    # Append centroid and distance to history
                    centroid_history.append({
                        "step": step,
                        "centroid": centroid.tolist(),
                        "distance_from_previous": distance
                    })

                    if distance < self.config['threshold']:
                        # Save final results specific to this material file
                        final_model_path = output_models_dir / f"{material_file.stem}_final.model"
                        self.train_word2vec(selected_papers, final_model_path)
                        break
                else:
                    # Append the initial centroid (no distance yet)
                    centroid_history.append({
                        "step": step,
                        "centroid": centroid.tolist(),
                        "distance_from_previous": None
                    })

                prev_centroid = centroid
                valid_centroid = True
                step += 1

            # Save centroid history for this material file
            centroid_history_file = centroid_history_dir / f"{material_file.stem}_centroid_history.csv"
            pd.DataFrame(centroid_history).to_csv(centroid_history_file, index=False)

        print("Workflow completed for all material files.")

def main():
    parser = argparse.ArgumentParser(description="Run the WorkflowPipeline.")
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
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the YAML config file."
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
    pipeline = WorkflowPipeline(config)
    pipeline.run_workflow(
        abstracts_csv=args.abstracts_csv,
        materials_dir=args.materials_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()