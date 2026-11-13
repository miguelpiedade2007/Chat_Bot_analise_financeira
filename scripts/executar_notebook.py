"""Executa o caderno completo e salva outputs reais, inclusive a célula em falha."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from app.config import ROOT


def main():
    caminho = ROOT / "notebooks" / "CKP02_DocMind_RAG.ipynb"
    notebook = nbformat.read(caminho, as_version=4)
    executor = NotebookClient(notebook, timeout=3600, kernel_name="python3",
                              resources={"metadata": {"path": str(ROOT)}})
    executor.on_cell_start = lambda cell, cell_index, **kw: print(
        f"Célula {cell_index + 1}/{len(notebook.cells)}: {cell.cell_type}", flush=True)
    try:
        executor.execute()
    finally:
        nbformat.write(notebook, caminho)
    print("Notebook executado integralmente sem erros.")


if __name__ == "__main__":
    main()
