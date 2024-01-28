import os
from pathlib import Path
import logging

logging.basicConfig(level= logging.INFO,format='[%(asctime)s]: %(message)s:')

list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "app.py",
    "stored_index.py",
    "static/.gitkeep",
    "templates/chat.html"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating the directory: {filedir} for the {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)):
        with open(filepath,'w') as f:
            pass
            logging.info(f'creating the empty file: {filepath}')
    else:
        logging.info(f'{filepath} already exists')