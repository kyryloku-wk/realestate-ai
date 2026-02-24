
Enable terminal

Here check python (very important, it should be not anaconda python)
# python --version 

Create new venv for project
# python -m venv .venv

Activate new env (using conda terminal)
# .venv\Scripts\activate

Install torch separatly with GPU
# pip install -U --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio

or with cpu
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Install venv
# pip install -e ".[dev]"


Kernel for Jupyter notebooks???
# python -m ipykernel install --user --name realestate-ai --display-name "realestate-ai"


Run docker services
# docker compose up -d
