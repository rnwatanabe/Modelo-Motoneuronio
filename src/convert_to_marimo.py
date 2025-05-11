import subprocess
from pathlib import Path


def convert_to_marimo():
    # Get all .ipynb files from backup directory
    notebooks = list(Path("backup_notebooks").glob("*.ipynb"))

    for notebook in notebooks:
        # Create output path with .py extension in the same directory as original
        output_path = notebook.parent.parent / notebook.stem
        output_path = output_path.with_suffix(".py")

        # Convert notebook to marimo format
        cmd = f'marimo convert "{notebook}" -o "{output_path}"'
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"Successfully converted {notebook} to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {notebook}: {e}")


if __name__ == "__main__":
    convert_to_marimo()
