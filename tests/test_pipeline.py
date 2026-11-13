"""Testes de comportamento sem chamadas pagas ou resultados RAGAS artificiais."""
import pytest
from pydantic import ValidationError
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from app.schemas import AnaliseConsulta, DocumentoFonte, RespostaRAG
from app import rag


def metadata():
    return dict(arquivo="teste.txt", fonte="https://example.org/fonte", titulo="Teste",
                pagina=1, tipo="txt", categoria="perfil", data="2026")


def test_validadores():
    with pytest.raises(ValidationError):
        AnaliseConsulta(tema="  ", categoria="geral", pontos_chave=["x"], nivel_confianca=.5)
    with pytest.raises(ValidationError):
        AnaliseConsulta(tema="BDR", categoria="geral", pontos_chave=[" "], nivel_confianca=.5)
    with pytest.raises(ValidationError):
        DocumentoFonte(**{**metadata(), "fonte": "sem origem"})
    with pytest.raises(ValidationError):
        RespostaRAG(resposta="x", fontes=[], contextos=["sem fonte"])


class EmbeddingTeste:
    """Vetores determinísticos exclusivamente para testar persistência e filtro."""
    def embed_documents(self, textos):
        return [[1., 0., 0.] for _ in textos]

    def embed_query(self, texto):
        return [1., 0., 0.]


def test_indexacao_recuperacao_e_cache(tmp_path):
    import chromadb
    cliente = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    docs = [Document(page_content="Texto de teste sobre perfil.", metadata=metadata())]
    emb = EmbeddingTeste()
    primeiro = rag.indexar(256, docs, emb, cliente)
    assert primeiro["chunks"] == 1
    assert rag.indexar(256, docs, emb, cliente)["cache"]
    assert rag.recuperar("perfil", 256, filtros={"categoria": "perfil"}, embeddings=emb, client=cliente)
    assert not rag.recuperar("perfil", 256, filtros={"categoria": "bdr"}, embeddings=emb, client=cliente)


@pytest.mark.parametrize("texto,aceita", [("Conceito [512_000001]", True),
                                            ("Conceito [512_000001, 512_999999]", False),
                                            ("Conceito [512_999999]", False),
                                            ("Resposta sem citação", False)])
def test_citacoes(monkeypatch, texto, aceita):
    trecho = dict(chunk_id="512_000001", texto="Conceito", distancia=0.,
                  metadata={**metadata(), "chunk_id": "512_000001"})
    monkeypatch.setattr(rag, "recuperar", lambda *a, **kw: [trecho])
    resultado = rag.responder("pergunta", llm=RunnableLambda(lambda prompt: texto))
    assert (resultado["resposta"] == texto) is aceita
    assert resultado["fontes"][0]["pagina"] == 1


def test_interface_inicia():
    from app.main import construir_interface
    assert construir_interface().blocks


def test_ragas_importa():
    from app.evaluation import avaliar_configuracoes
    assert callable(avaliar_configuracoes)


def test_titulo_explicito_restringe_fonte(monkeypatch, tmp_path):
    import chromadb
    cliente = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    meta = metadata()
    docs = [Document(page_content="Cuidados educativos", metadata=meta),
            Document(page_content="Outro documento", metadata={**meta, "arquivo": "outro.txt"})]
    emb = EmbeddingTeste()
    rag.indexar(256, docs, emb, cliente)
    monkeypatch.setattr(rag, "ler_manifesto", lambda: {"teste.txt": {"titulo": "CVM: Investidor 60+"}})
    resultado = rag.recuperar("O que diz o guia Investidor 60+?", 256, embeddings=emb, client=cliente)
    assert {r["metadata"]["arquivo"] for r in resultado} == {"teste.txt"}
