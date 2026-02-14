
Enable terminal

Here check python
# python --version

Create new venv for project
# python -m venv .venv

Activate new env (using conda terminal)
# .venv\Scripts\activate

Install torch separatly with GPU
# pip install torch --index-url https://download.pytorch.org/whl/cu121

or with cpu
# pip install torch --index-url https://download.pytorch.org/whl/cpu

Install venv
# pip install -e ".[dev]"


Kernel for Jupyter notebooks???
# python -m ipykernel install --user --name realestate-ai --display-name "realestate-ai"


Run docker services
# docker compose up -d
