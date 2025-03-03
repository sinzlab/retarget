import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config
