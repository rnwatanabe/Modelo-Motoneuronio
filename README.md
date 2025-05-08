# Modelo-Motoneurônio

Rebeka Batichotti

Renato Watanabe

## Preparando o ambiente

### No Linux

Executar as seguintes instruções:

`python -m venv modelpynn`

`source modelpynn/bin/activate`

`pip install -r requirements.txt`

### No Windows

Executar as seguintes instruções:

`python -m venv modelpynn`

`.\modelpynn\Scripts\activate`

`pip install -r requirements.txt`


## Instalação do Neuron

Instalar o Neuron separadamente. Como no Windows ele não pode ser instalado via pip, ele não foi incluído no arquivo requirements.txt. 

### No Linux

digitar no terminal:

`pip install neuron`

### No Windows

Instalar o Neuron com o instalador em [https://github.com/neuronsimulator/nrn/releases/download/8.2.0/nrn-8.2.0.w64-mingw-py-37-38-39-310-setup.exe](https://github.com/neuronsimulator/nrn/releases/download/8.2.0/nrn-8.2.0.w64-mingw-py-37-38-39-310-setup.exe).

Após a instalação o VS Code deve ser reinicializado para que variável de ambiente PYTHONPATH tenha o caminho do Neuron.

## Teste

Por enquanto, a melhor maneira de testar é executar todas as células do arquivo [Notebooks/testew_Ca_delays.ipynb](Notebooks/testew_Ca_delays.ipynb). Pegar a última versão do arquivo.

