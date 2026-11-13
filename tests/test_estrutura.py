"""Testes sem chave/API: validam arquivos, manifestos e sintaxe do projeto."""
import ast
import json
from pathlib import Path
from app.config import ROOT, chunk_config, CHUNK_SIZES, SEPARATORS


def test_python_sintaxe():
    for arquivo in ROOT.rglob("*.py"):
        ast.parse(arquivo.read_text(encoding="utf-8"), filename=str(arquivo))


def test_chunking_exigido():
    assert CHUNK_SIZES == (256, 512)
    for size in CHUNK_SIZES:
        cfg = chunk_config(size)
        assert cfg["separators"] == ["\n\n", "\n", ". ", " ", ""]
        assert 0.10 <= cfg["chunk_overlap"] / size <= 0.15


def test_fontes_reais_identificadas():
    fontes = json.loads((ROOT / "data" / "fontes.json").read_text(encoding="utf-8"))
    assert len(fontes) >= 5
    assert len(set(f["arquivo"] for f in fontes)) == len(fontes)
    assert all(f["url"].startswith("https://www.gov.br/") for f in fontes)
    assert all(f["arquivo"].endswith(".pdf") for f in fontes)
    assert all(f["titulo"] and f["categoria"] for f in fontes)


def test_perguntas():
    perguntas = json.loads((ROOT / "data" / "perguntas_teste.json").read_text(encoding="utf-8"))
    assert len(perguntas) >= 5
    assert len(perguntas) == len(set(perguntas))


def test_notebook_valido():
    import nbformat
    caminho = ROOT / "notebooks" / "CKP02_DocMind_RAG.ipynb"
    nbformat.validate(nbformat.read(caminho, as_version=4))


def test_documentos_exigem_fonte(monkeypatch, tmp_path):
    """Sem PDFs reais, simulamos um arquivo apenas nos testes (não na base entregue)."""
    import importlib
    mod = importlib.import_module("app.documentos")
    pasta = tmp_path / "documentos"
    pasta.mkdir()
    (pasta / "sem_fonte.md").write_text("Texto fictício somente de teste.", encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="sem URL de origem"):
        mod.carregar_documentos(pasta, minimo=1)
