
# import libraries
import os
import logging
from pathlib import Path

# set up logging string
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# set project name
project_name = "kidney_disease_classifier"

# specify files and folders to be created
list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "Dockerfile"
    "requirements.txt",
    "setup.py",
    "streamlit.py",
    "research/trials.ipynb",
    "templates/index.html",

]

# Iterate over each file path in the list of files
for filepath in list_of_files:

    # Convert the filepath string to a Path object for easier manipulation
    filepath = Path(filepath)

    # Split the filepath into its directory and filename components
    filedir, filename = os.path.split(filepath)

    # Check if the directory part of the filepath is not empty
    if filedir !="":

        # Create the directory if it doesn't exist, without raising an error if it does
        os.makedirs(filedir, exist_ok=True)

        # log the meaasge that the file and directory created
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    # Check if the file does not exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        
        # Open the file in write mode (this creates the file if it doesn't exist)
        with open(filepath, "w") as f:

            # No content is written to the file
            pass

            # log the message that created an empty firl    
            logging.info(f"Creating empty file: {filepath}")

    else:

        # Log that the file already exists
        logging.info(f"{filename} is already exists")