import yaml


class Config:
    """
    Configuration class to load and manage configuration settings from a YAML file.
    """

    def __init__(self, config_path: str, section: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.config = config[section]
        self.__dict__.update(self.config)
