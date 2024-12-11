import json
import requests # type: ignore
import logging
import json
from typing import Union
import matplotlib.pyplot as plt # type: ignore

logging.basicConfig(level=logging.INFO)

def get_key(key_filename: str) -> str:
    with open(key_filename, 'r') as file:
        lines = file.readlines()
        return str(lines[0])

def write_to_json_file(filename: str, output_file_dir: str, data: dict) -> None:
    with open(f'{filename}', 'w') as json_file:
        json.dump(data, json_file, indent=4)
