"""Monta ZIP com lista explícita de pastas, sem .env, caches ou banco gerado."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from app.config import ROOT


def main():
    from scripts.verificar_entrega import main as verificar
    verificar()
    destino = ROOT.parent / "AnalistaBot_CKP02_entrega.zip"
    pastas = ("app", "scripts", "tests", "notebooks", "resultados", "data/documentos")
    arquivos = [ROOT / nome for nome in (
        "README.md", "RELATORIO_REVISAO.md", ".env.example", ".gitignore", "requirements.txt",
        "requirements-rerank.txt", "VERSOES_TESTADAS.txt", "context_rot.py",
        "data/fontes.json", "data/perguntas_teste.json")]
    for pasta in pastas:
        arquivos.extend((ROOT / pasta).rglob("*"))
    with ZipFile(destino, "w", ZIP_DEFLATED) as zipado:
        for arquivo in sorted(set(arquivos)):
            if arquivo.is_file() and "__pycache__" not in arquivo.parts and arquivo.suffix != ".pyc":
                zipado.write(arquivo, Path(ROOT.name) / arquivo.relative_to(ROOT))
    with ZipFile(destino) as zipado:
        assert not any(Path(nome).name == ".env" for nome in zipado.namelist())
        assert zipado.testzip() is None
        from dotenv import dotenv_values
        segredo = dotenv_values(ROOT / ".env").get("OLLAMA_API_KEY")
        if segredo:
            assert not any(segredo.encode() in zipado.read(nome) for nome in zipado.namelist()), "Chave encontrada no pacote"
    print(f"Pacote criado: {destino.name} ({destino.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
