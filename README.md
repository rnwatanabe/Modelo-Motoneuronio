# Modelo-Motoneurônio

Rebeka Batichotti

Renato Watanabe

## Prepare the environment

### On Linux

Execute the following instructions:

`python -m venv modelpynn`

`source modelpynn/bin/activate`

`pip install -r requirements.txt`

### On Windows

Execute the following instructions:

`python -m venv modelpynn`

`.\modelpynn\Scripts\activate`

`pip install -r requirements.txt`

## Install Neuron

Install Neuron separately. As it cannot be installed via pip on Windows, it was not included in the requirements.txt file.

### On Linux

Type in the terminal:

`pip install neuron`

### On Windows

Install Neuron with the installer at [https://github.com/neuronsimulator/nrn/releases/download/8.2.0/nrn-8.2.0.w64-mingw-py-37-38-39-310-setup.exe](https://github.com/neuronsimulator/nrn/releases/download/8.2.0/nrn-8.2.0.w64-mingw-py-37-38-39-310-setup.exe).

After installation, VS Code should be restarted so that the PYTHONPATH environment variable has the Neuron path.

## Teste

For now, the best way to test is to execute all the cells of the file [Notebooks/testew_Ca_delays.ipynb](Notebooks/testew_Ca_delays.ipynb). Get the latest version of the file.

