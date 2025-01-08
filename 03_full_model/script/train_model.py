
import argparse
import yaml
import sys
import pandas as pd
from pathlib import Path
from matnexus import VecGenerator


class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def train_word2vec(self, abstracts_csv, model_output):
        """
        Train a Word2Vec model on abstracts and save the model.

        :param abstracts_csv: Path to the CSV file containing abstracts
        :param model_output:  Path to save the trained Word2Vec model
        """
        # Read abstracts
        abstracts_df = pd.read_csv(abstracts_csv)

        # Retrieve parameters from config
        params = self.config["word2vec"]

        # Convert DataFrame into sentences using matnexus
        corpus = VecGenerator.Corpus(abstracts_df)
        sentences = corpus.sentences

        # Create and fit the model
        model = VecGenerator.Word2VecModel(sentences)
        model.fit(
            sg=params.get("sg", 1),
            vector_size=params.get("vector_size", 100),
            hs=params.get("hs", 0),
            window=params.get("window", 5),
            min_count=params.get("min_count", 1),
            workers=params.get("workers", 4)
        )

        # Save model
        model.save(str(model_output))
        print(f"Model saved to {model_output}")


def main():
    parser = argparse.ArgumentParser(description="Train a Word2Vec model and save it.")
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
        "--model_output",
        type=str,
        required=True,
        help="Path (including filename) to save the trained model."
    )
    args = parser.parse_args()

    # Read config from YAML file
    config_path = Path(args.config_file)
    if not config_path.is_file():
        print(f"Error: config file {config_path} not found.")
        sys.exit(1)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Train the model
    trainer = ModelTrainer(config)
    trainer.train_word2vec(abstracts_csv=args.abstracts_csv,
                           model_output=args.model_output)


if __name__ == "__main__":
    main()