import json

def load_config():
    with open('config.json', 'r') as config_file:
        return json.load(config_file)

# Load configuration once
config = load_config()

# Dynamically set variables based on the config keys
for section, settings in config.items():
    for key, value in settings.items():
        globals()[key] = value
