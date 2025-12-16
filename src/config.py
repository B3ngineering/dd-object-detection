import yaml
from pathlib import Path

def get_config(config_path='src/config.yaml') -> dict:
    """Load configuration from a YAML file."""

    path = Path(config_path)
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_model_configs(config) -> list:
    model_configs = []

    for n, settings in config["models"].items():
        model_config = {
            "path": settings["path"],
            "weight": settings.get("weight", 1.0),
            "priority": settings.get("priority", 0)
        }
        model_configs.append(model_config)

    return model_configs
