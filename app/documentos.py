"""Load de documentos reais com identificação verificável da origem."""
import json
from pathlib import Path
from langchain_core.documents import Document
from pypdf import PdfReader
from app.config import DOCUMENTOS, FONTES_MANIFESTO
from app.schemas import DocumentoFonte

EXTENSOES = {".pdf", ".txt", ".md"}


def ler_manifesto():
    if not FONTES_MANIFESTO.is_file():
        return {}
    registros = json.loads(FONTES_MANIFESTO.read_text(encoding="utf-8"))
    return {registro["arquivo"]: registro for registro in registros}


def carregar_documentos(pasta=DOCUMENTOS, minimo=5):
    """Carrega PDF por página (citação precisa) e TXT/MD como texto integral."""
    pasta = Path(pasta)
    if not pasta.is_dir():
        raise FileNotFoundError(f"Pasta de documentos inexistente: {pasta}")
    arquivos = sorted(f for f in pasta.iterdir() if f.suffix.lower() in EXTENSOES)
    if len(arquivos) < minimo:
        raise ValueError(
            f"Base incompleta: {len(arquivos)} documento(s); mínimo {minimo}. "
            "Execute 'python -m scripts.baixar_fontes' ou adicione documentos reais."
        )
    registro = ler_manifesto()
    documentos = []
    for caminho in arquivos:
        inicio_documento = len(documentos)
        if caminho.name not in registro or not registro[caminho.name].get("url"):
            raise ValueError(f"Documento sem URL de origem em data/fontes.json: {caminho.name}")
        info = registro[caminho.name]
        metadados = {
            "arquivo": caminho.name,
            "fonte": str(info.get("url") or caminho.resolve().as_uri()),
            "tipo": str(info.get("tipo") or caminho.suffix[1:].upper()),
            "categoria": str(info.get("categoria") or "acoes"),
            "data": str(info.get("data") or "nao_informada"),
            "titulo": str(info.get("titulo") or caminho.stem),
        }
        if caminho.suffix.lower() == ".pdf":
            try:
                pdf = PdfReader(str(caminho))
                for n, pagina in enumerate(pdf.pages, 1):
                    texto = (pagina.extract_text() or "").strip()
                    if texto:
                        meta = DocumentoFonte(**metadados, pagina=n).model_dump()
                        documentos.append(Document(page_content=texto, metadata=meta))
            except Exception as exc:
                raise RuntimeError(f"Falha ao ler PDF real {caminho.name}: {exc}") from exc
        else:
            texto = caminho.read_text(encoding="utf-8").strip()
            if texto:
                meta = DocumentoFonte(**metadados, pagina=1).model_dump()
                documentos.append(Document(page_content=texto, metadata=meta))
        if len(documentos) == inicio_documento:
            raise ValueError(f"Documento sem texto extraível: {caminho.name}. Faça OCR antes de indexar.")
    if not documentos:
        raise ValueError("Documentos não possuem texto extraível; verifique PDFs escaneados.")
    return documentos
