"""Baixa PDFs originais diretamente do domínio gov.br/CVM; não gera dados fictícios."""
import json
import sys
from pathlib import Path
import requests
from app.config import FONTES_MANIFESTO, DOCUMENTOS


def baixar_fontes():
    DOCUMENTOS.mkdir(parents=True, exist_ok=True)
    fontes = json.loads(FONTES_MANIFESTO.read_text(encoding="utf-8"))
    falhas = []
    for fonte in fontes:
        destino = DOCUMENTOS / fonte["arquivo"]
        if destino.is_file() and destino.read_bytes()[:4] == b"%PDF":
            print(f"[já existe] {destino.name}")
            continue
        try:
            resposta = requests.get(
                fonte["url"], timeout=80,
                headers={"User-Agent": "Mozilla/5.0 (academico; projeto de RAG)"},
            )
            resposta.raise_for_status()
            if not resposta.content.startswith(b"%PDF"):
                raise ValueError("O servidor não devolveu um PDF válido")
            destino.write_bytes(resposta.content)
            print(f"[baixado] {destino.name} ({len(resposta.content):,} bytes)")
        except Exception as exc:
            print(f"[FALHA] {destino.name}: {exc}", file=sys.stderr)
            falhas.append(destino.name)
    if falhas:
        raise RuntimeError("Falhou o download de: " + ", ".join(falhas) + ". Baixe pelos URLs em data/fontes.json.")
    return len(fontes)


if __name__ == "__main__":
    print(f"Fontes oficiais baixadas: {baixar_fontes()}")
