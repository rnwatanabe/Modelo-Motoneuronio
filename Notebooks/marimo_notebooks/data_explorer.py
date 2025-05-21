#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import marimo as mo
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

_ = mo.md("""
# Motoneuron Data Explorer

This app allows you to explore the data in the data folder. The data includes:

- **FDI**: Experiments with iEMG in the FDI muscle while applying force at different angles with the index finger (FSAMP = 10240, RefSignal is Force in N)
- **TA**: FSAMP = 10240, RefSignal is % MVC
- **POx_iEMG_Ramps**: FSAMP = 10240, RefSignal is % MVC
  - 3 force levels 5, 15, 30% MVC
  - RampDefinition_xx.mat has the start and stop of each ramp
  - VL = Vastus Lateralis
  - VM = Vastus Medialis
  - Only trust plateau phase (use ramp definitions). Up and Down phases are not super cleaned
""")


@mo.cell
def _(mo=mo):
    # Function to list all available data folders
    def get_data_folders():
        data_path = Path("data")
        folders = [f for f in data_path.iterdir() if f.is_dir()]
        return folders
    
    data_folders = get_data_folders()
    
    folder_selector = mo.ui.dropdown(
        options=[str(folder.name) for folder in data_folders],
        value=str(data_folders[0].name) if data_folders else None,
        label="Select Data Folder"
    )
    
    return folder_selector


@mo.cell
def _(folder_selector, mo=mo):
    # Function to list all available data subfolders
    def get_subfolders(folder_name):
        folder_path = Path("data") / folder_name
        subfolders = [f for f in folder_path.iterdir() if f.is_dir()]
        return subfolders
    
    subfolders = get_subfolders(folder_selector)
    
    subfolder_selector = mo.ui.dropdown(
        options=[str(subfolder.name) for subfolder in subfolders],
        value=str(subfolders[0].name) if subfolders else None,
        label="Select Subfolder"
    )
    
    return subfolder_selector


@mo.cell
def _(folder_selector, subfolder_selector, mo=mo):
    # Function to list all available .mat files
    def get_mat_files(folder_name, subfolder_name):
        folder_path = Path("data") / folder_name / subfolder_name
        mat_files = [f for f in folder_path.iterdir() if f.suffix == '.mat']
        return mat_files
    
    mat_files = get_mat_files(folder_selector, subfolder_selector)
    
    file_selector = mo.ui.dropdown(
        options=[str(file.name) for file in mat_files],
        value=str(mat_files[0].name) if mat_files else None,
        label="Select .mat File"
    )
    
    return file_selector


@mo.cell
def _(folder_selector, subfolder_selector, file_selector, mo=mo):
    if not file_selector:
        return mo.md("No file selected")
    
    file_path = Path("data") / folder_selector / subfolder_selector / file_selector
    
    # Load the .mat file
    try:
        mat_data = sio.loadmat(file_path)
        
        # Display the keys in the .mat file
        keys = list(mat_data.keys())
        # Filter out keys that start with '__' (metadata)
        data_keys = [key for key in keys if not key.startswith('__')]
        
        key_selector = mo.ui.dropdown(
            options=data_keys,
            value=data_keys[0] if data_keys else None,
            label="Select Data Variable"
        )
        
        return key_selector, mat_data
    except Exception as e:
        return mo.md(f"Error loading file: {e}")


@mo.cell
def _(key_selector, mat_data, mo=mo):
    if not isinstance(key_selector, tuple) or key_selector[0] is None:
        return mo.md("No data variable selected")
    
    selected_key = key_selector[0]
    data = mat_data[selected_key]
    
    # Display basic information about the data
    info_text = f"""
    ## Data Information
    
    - **Variable Name**: {selected_key}
    - **Data Type**: {type(data).__name__}
    - **Shape**: {data.shape if hasattr(data, 'shape') else 'N/A'}
    """
    
    # If data is a structured array, show field names
    if hasattr(data, 'dtype') and data.dtype.names is not None:
        info_text += f"- **Fields**: {', '.join(data.dtype.names)}\n"
    
    return mo.md(info_text), data


@mo.cell
def _(data, mo=mo):
    # Function to plot the data
    if not isinstance(data, tuple) or data[1] is None:
        return mo.md("No data to plot")
    
    selected_data = data[1]
    
    # Create plot options based on data type
    plot_options = ["Time Series"]
    
    if hasattr(selected_data, 'shape') and len(selected_data.shape) == 2:
        plot_options.extend(["Heatmap", "Histogram"])
    
    plot_type = mo.ui.dropdown(
        options=plot_options,
        value=plot_options[0],
        label="Select Plot Type"
    )
    
    return plot_type, selected_data


@mo.cell
def _(plot_type, mo=mo):
    if not isinstance(plot_type, tuple) or plot_type[0] is None:
        return mo.md("No plot type selected")
    
    selected_plot_type = plot_type[0]
    selected_data = plot_type[1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        if selected_plot_type == "Time Series":
            # If data is 1D, plot directly
            if len(selected_data.shape) == 1:
                ax.plot(selected_data)
                ax.set_title("Time Series Plot")
                ax.set_xlabel("Sample Index")
                ax.set_ylabel("Value")
            # If data is 2D, plot first few rows
            elif len(selected_data.shape) == 2:
                num_rows_to_plot = min(5, selected_data.shape[0])
                for i in range(num_rows_to_plot):
                    ax.plot(selected_data[i], label=f"Series {i+1}")
                ax.legend()
                ax.set_title(f"Time Series Plot (First {num_rows_to_plot} series)")
                ax.set_xlabel("Sample Index")
                ax.set_ylabel("Value")
        
        elif selected_plot_type == "Heatmap":
            # Plot as heatmap
            sns.heatmap(selected_data, cmap="viridis", ax=ax)
            ax.set_title("Heatmap")
        
        elif selected_plot_type == "Histogram":
            # Plot histogram
            ax.hist(selected_data.flatten(), bins=50)
            ax.set_title("Histogram")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
    
    except Exception as e:
        plt.close(fig)
        return mo.md(f"Error plotting data: {e}")
    
    return mo.ui.plotly(fig)


@mo.cell
def _(mo=mo):
    return mo.md("""
    ## Data Export Options
    
    You can export the selected data to CSV or Excel format.
    """)


@mo.cell
def _(data, mo=mo):
    if not isinstance(data, tuple) or data[1] is None:
        return mo.md("No data to export")
    
    selected_data = data[1]
    
    export_format = mo.ui.dropdown(
        options=["CSV", "Excel"],
        value="CSV",
        label="Export Format"
    )
    
    export_filename = mo.ui.text(
        value="exported_data",
        label="Filename (without extension)"
    )
    
    export_button = mo.ui.button(label="Export Data")
    
    return export_format, export_filename, export_button, selected_data


@mo.cell
def _(export_format, export_filename, export_button, mo=mo):
    if not isinstance(export_format, tuple):
        return mo.md("Export options not configured")
    
    format_choice = export_format[0]
    filename = export_format[1]
    button_clicked = export_format[2]
    selected_data = export_format[3]
    
    if not button_clicked.value:
        return mo.md("Click the Export Data button to export")
    
    try:
        # Convert data to DataFrame if possible
        if hasattr(selected_data, 'shape'):
            if len(selected_data.shape) == 1:
                df = pd.DataFrame(selected_data, columns=["Value"])
            else:
                df = pd.DataFrame(selected_data)
        else:
            return mo.md("Data cannot be converted to DataFrame for export")
        
        # Export based on selected format
        if format_choice == "CSV":
            output_path = f"{filename}.csv"
            df.to_csv(output_path, index=False)
        else:  # Excel
            output_path = f"{filename}.xlsx"
            df.to_excel(output_path, index=False)
        
        return mo.md(f"Data exported successfully to {output_path}")
    
    except Exception as e:
        return mo.md(f"Error exporting data: {e}")
