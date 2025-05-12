# Modelo-Motoneurônio

[![Python](https://img.shields.io/badge/Python->=_3.10-blue.svg)](https://www.python.org/)
[![NEURON](https://img.shields.io/badge/NEURON->=_8.2.0-orange.svg)](https://neuron.yale.edu/neuron/)

## 👥 Contributors
- **Rebeka Batichotti**
- **Renato Watanabe**

## 📋 Overview
*A computational model for motoneuron simulation.*

## ⚙️ Setup Environment

### 1. Install uv (Dependency Manager)

Get the latest version of uv from [the official documentation](https://docs.astral.sh/uv/getting-started/installation/).

<details open>
<summary>💻 <b>Windows Installation</b></summary>

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
</details>

<details>
<summary>🐧 <b>Linux/MacOS Installation</b></summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
</details>

### 2. Install Dependencies

```bash
uv sync
```

#### 2.1 Windows NEURON Installation

> [!IMPORTANT]     
> On Windows, NEURON must be installed separately as it cannot be installed via pip.  
>
> - Download from [NEURON Documentation](https://nrn.readthedocs.io/en/8.2.6/install/install_instructions.html)
> - Install to the default directory: `C:\nrn`

Configure the `PYTHONPATH` environment variable for your IDE:

| IDE    | Configuration                                               |
|--------|-------------------------------------------------------------|
| VSCode | PYTHONPATH environment variable preset automatically        |
| Others | Set PYTHONPATH to: `C:\nrn\lib\python`                      |

## 📝 Run Marimo Notebooks

```bash
uv run marimo edit Notebooks/marimo_notebooks/NOTEBOOK_NAME.py
```

