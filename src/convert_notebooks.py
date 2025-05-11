import os
import subprocess
from pathlib import Path


def convert_notebooks():
    # Get all .ipynb files recursively
    notebooks = list(Path(".").rglob("*.ipynb"))

    for notebook in notebooks:
        # Create output path with .py extension
        output_path = notebook.with_suffix(".py")

        # Convert notebook to Python script
        cmd = (
            f'jupyter nbconvert --to python "{notebook}" --output "{output_path.stem}"'
        )
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"Successfully converted {notebook} to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {notebook}: {e}")


if __name__ == "__main__":
    convert_notebooks()
