import argparse
import yaml


def get_config():
    parser = argparse.ArgumentParser(description="Parse configuration file")
    parser.add_argument(
        "config_file", type=str, help="Path to the configuration yaml file"
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    return config


def load_input_features(input_features_file):
    with open(input_features_file, "r") as fh:
        input_features = yaml.safe_load(fh)
    return input_features
