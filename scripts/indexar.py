"""Cria duas coleções persistentes no ChromaDB para comparar os chunkings."""
from app.config import CHUNK_SIZES
from app.documentos import carregar_documentos
from app.rag import indexar, build_embeddings


def main():
    docs = carregar_documentos(minimo=5)
    embedding = build_embeddings()
    print(f"Teste embed_query(): {len(embedding.embed_query('O que é uma ação?'))} dimensões")
    print(f"Teste embed_documents(): {len(embedding.embed_documents(['Mercado de capitais']))} vetor")
    for tamanho in CHUNK_SIZES:
        print(indexar(tamanho, documentos=docs, embeddings=embedding))


if __name__ == "__main__":
    main()
