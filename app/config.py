"""Constantes e caminhos compartilhados; sem chamadas externas ao importar."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DOCUMENTOS = ROOT / "data" / "documentos"
CHROMA_DIR = ROOT / "data" / "chroma"
RESULTADOS = ROOT / "resultados"
FONTES_MANIFESTO = ROOT / "data" / "fontes.json"
DOMINIO = "analise_de_acoes"
CHUNK_SIZES = (256, 512)
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
CHAT_MODEL = "gemma4:cloud"
EMBED_MODEL = "nomic-embed-text"


def embedding_base_url():
    return os.getenv("OLLAMA_EMBED_BASE_URL", "http://localhost:11434").rstrip("/")


def chunk_config(size):
    """Overlap de 12,5% do tamanho, intervalo exigido: 10–15%."""
    if size <= 0:
        raise ValueError("chunk_size deve ser positivo")
    return {"chunk_size": size, "chunk_overlap": round(size * 0.125), "separators": SEPARATORS}


def tamanho_selecionado():
    """Usa a comparação real quando disponível; permite escolha explícita no .env."""
    import json
    escolhido = os.getenv("RAG_CHUNK_SIZE", "").strip()
    arquivo = RESULTADOS / "configuracao_vencedora.json"
    tamanho = int(escolhido) if escolhido else (
        int(json.loads(arquivo.read_text(encoding="utf-8"))["chunk_size"]) if arquivo.exists() else 512)
    if tamanho not in CHUNK_SIZES:
        raise ValueError(f"RAG_CHUNK_SIZE precisa ser uma das configurações comparadas: {CHUNK_SIZES}")
    return tamanho
