# Motoneuron Data Explorer

This is a marimo app for exploring the motoneuron data in the data folder.

## Requirements

- Python 3.8+
- marimo
- numpy
- scipy
- matplotlib
- pandas
- seaborn

## Installation

If you don't have marimo installed, you can install it with:

```bash
pip install marimo
```

You'll also need to install the other dependencies:

```bash
pip install numpy scipy matplotlib pandas seaborn
```

## Running the App

To run the app, use the following command:

```bash
marimo run data_explorer.py
```

Or to run it in edit mode:

```bash
marimo edit data_explorer.py
```

## Features

The app allows you to:

1. Browse through the data folders and files
2. Select and load .mat files
3. Explore the variables within the .mat files
4. Visualize the data with different plot types:
   - Time series
   - Heatmap
   - Histogram
5. Export the data to CSV or Excel format

## Data Structure

According to the README.txt in the data folder:

- **FDI**: Experiments with iEMG in the FDI muscle while applying force at different angles with the index finger
  - FSAMP = 10240
  - RefSignal is Force in N

- **TA**:
  - FSAMP = 10240
  - RefSignal is % MVC

- **POx_iEMG_Ramps**:
  - FSAMP = 10240
  - RefSignal is % MVC
  - 3 force levels 5, 15, 30% MVC
  - RampDefinition_xx.mat has the start and stop of each ramp
  - VL = Vastus Lateralis
  - VM = Vastus Medialis
  - Only trust plateau phase (use ramp definitions). Up and Down phases are not super cleaned
