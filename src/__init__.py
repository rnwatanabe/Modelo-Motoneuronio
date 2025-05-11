"""
Initialize and set up NMODL (NEURON MODeling Language) files for the model.

This module handles the compilation and loading of NMODL files, which are used to define
custom mechanisms and models in NEURON simulations. It performs the following steps:
1. Locates and copies NMODL files to the appropriate directory
2. Compiles the NMODL files (platform-specific approach)
3. Loads the compiled files into NEURON

The module is automatically executed when the package is imported.
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

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

def find_mknrndll() -> Optional[Path]:
    """Find the mknrndll executable on Windows systems."""
    # Common locations for mknrndll
    possible_locations = [
        Path(os.environ.get('NEURONHOME', '')) / 'bin',
        Path(os.environ.get('NEURONHOME', '')) / 'mingw',
        Path("C:/nrn/bin"),
        Path("C:/Program Files/NEURON/bin"),
        Path("C:/Program Files (x86)/NEURON/bin"),
    ]

    for location in possible_locations:
        mknrndll_path = location / "mknrndll.bat"
        if mknrndll_path.exists():
            return mknrndll_path

    # Try to find it in PATH
    try:
        result = subprocess.run(["where", "mknrndll.bat"],
                              capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except Exception:
        pass

    return None

def compile_mod_files_windows(nmodl_path: Path) -> None:
    """Compile NMODL files on Windows using mknrndll."""
    mknrndll_path = find_mknrndll()

    if mknrndll_path is None:
        raise FileNotFoundError(
            "Could not find mknrndll.bat. Please make sure NEURON is properly installed "
            "and NEURONHOME environment variable is set correctly."
        )

    print(f"Using mknrndll: {mknrndll_path}")

    # Change to the directory containing the mod files and run mknrndll.bat
    original_dir = os.getcwd()
    try:
        os.chdir(nmodl_path)
        # On Windows, we need to use cmd.exe to run batch files
        cmd = ["cmd", "/c", str(mknrndll_path)]
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error during compilation: {e.stderr}")
        raise
    finally:
        os.chdir(original_dir)

def compile_mod_files_unix(nmodl_path: Path) -> None:
    """Compile NMODL files on Unix-like systems using pyNN's utility."""
    from pyNN.utility.build import compile_nmodl
    try:
        print(f"Compiling NMODL files from {nmodl_path}")
        compile_nmodl(nmodl_path)
    except Exception as e:
        print(f"Error: Failed to compile NMODL files: {str(e)}")
        raise

def compile_and_load_mod_files(nmodl_path: Path, mod_files: List[Path]) -> None:
    """Compile and load NMODL files into NEURON based on platform."""
    if not mod_files:
        print("No mod files to compile")
        return

    # Platform-specific compilation
    if platform.system() == 'Windows':
        compile_mod_files_windows(nmodl_path)
    else:
        compile_mod_files_unix(nmodl_path)

    # Load the compiled mechanisms
    # The location and naming of compiled files differs between platforms
    if platform.system() == 'Windows':
        # On Windows, NEURON creates a 'nrnmech.dll' file
        dll_path = nmodl_path / "nrnmech.dll"
        if dll_path.exists():
            try:
                h.nrn_load_dll(str(dll_path))
                print(f"Successfully loaded nrnmech.dll")
            except Exception as e:
                print(f"Warning: Error loading nrnmech.dll: {str(e)}")
                print("This may be because some mechanisms are already loaded, which is usually not a problem.")
                # Continue execution - don't re-raise the exception
        else:
            print(f"Warning: Expected {dll_path} was not found after compilation")
    else:
        # On Unix, load individual .o files
        for mod_file in mod_files:
            o_file_path = str(nmodl_path / f"{mod_file.stem}.o")
            try:
                print(f"Loading {o_file_path}")
                h.nrn_load_dll(o_file_path)
                print(f"Successfully loaded {mod_file.stem}")
            except Exception as e:
                print(f"Warning: Failed to load {mod_file.stem}: {str(e)}")
                print("This may be because the mechanism is already loaded, which is usually not a problem.")

def main():
    """Main function to handle NMODL file setup."""
    try:
        src_path = Path(__file__).parent
        print(f"Loading NMODL files from {src_path}")

        nmodl_path = find_nmodl_directory()
        mod_files = copy_mod_files(src_path, nmodl_path)

        if mod_files:
            compile_and_load_mod_files(nmodl_path, mod_files)
            print("NMODL files processing complete!")
        else:
            print("Warning: No NMODL files were processed")

    except Exception as e:
        print(f"Error during NMODL setup: {str(e)}")
        # Log the error but don't crash the program
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
else:
    # When imported as a module
    main()
