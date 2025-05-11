"""
Initialize and set up NMODL (NEURON MODeling Language) files for the model.

This module handles the compilation and loading of NMODL files, which are used to define
custom mechanisms and models in NEURON simulations. It performs the following steps:
1. Locates and copies NMODL files to the appropriate directory
2. Compiles the NMODL files
3. Loads the compiled files into NEURON

The module is automatically executed when the package is imported.
"""

import shutil
from pathlib import Path
from typing import List

from pyNN.utility.build import compile_nmodl
from neuron import h

def find_nmodl_directory() -> Path:
    """Find the NMODL directory in the pyNN/neuron installation."""
    src_path = Path(__file__).parent
    try:
        nmodl_path = next(src_path.parent.rglob("*pyNN/neuron/nmodl"))
        return nmodl_path
    except StopIteration:
        print("Error: Could not find pyNN/neuron/nmodl directory")
        raise FileNotFoundError("Could not find pyNN/neuron/nmodl directory")

def copy_mod_files(src_path: Path, nmodl_path: Path) -> List[Path]:
    """Copy .mod files from source to NMODL directory."""
    mod_files = list(src_path.glob("*mod"))
    if not mod_files:
        print("Warning: No .mod files found in source directory")
        return []
    
    for mod_file in mod_files:
        try:
            shutil.copy(mod_file, nmodl_path / mod_file.name)
            print(f"Copied {mod_file.stem}")
        except (shutil.Error, IOError) as e:
            print(f"Error: Failed to copy {mod_file.stem}: {str(e)}")
            raise
    
    return mod_files

def compile_and_load_mod_files(nmodl_path: Path, mod_files: List[Path]) -> None:
    """Compile and load NMODL files into NEURON."""
    try:
        print(f"Compiling NMODL files from {nmodl_path}")
        compile_nmodl(nmodl_path)
    except Exception as e:
        print(f"Error: Failed to compile NMODL files: {str(e)}")
        raise

    for mod_file in mod_files:
        o_file_path = str(nmodl_path / f"{mod_file.stem}.o")
        try:
            print(f"Loading {o_file_path}")
            h.nrn_load_dll(o_file_path)
            print(f"Successfully loaded {mod_file.stem}")
        except Exception as e:
            print(f"Error: Failed to load {mod_file.stem}: {str(e)}")
            raise

def main():
    """Main function to handle NMODL file setup."""
    try:
        src_path = Path(__file__).parent
        print(f"Loading NMODL files from {src_path}")
        
        nmodl_path = find_nmodl_directory()
        mod_files = copy_mod_files(src_path, nmodl_path)
        
        if mod_files:
            compile_and_load_mod_files(nmodl_path, mod_files)
            print("All NMODL files successfully loaded!")
        else:
            print("Warning: No NMODL files were processed")
            
    except Exception as e:
        print(f"Error: Failed to initialize NMODL files: {str(e)}")
        raise

if __name__ == "__main__":
    main()
else:
    # When imported as a module
    main()
