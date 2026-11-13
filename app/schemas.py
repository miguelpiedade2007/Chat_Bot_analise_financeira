"""
Schemas Pydantic v2 — validam a saída estruturada do chatbot.

Os nomes/campos abaixo já estão no vocabulário de "análise de ações", mas
a estrutura (BaseModel com 4+ campos tipados) é genérica: se o domínio
mudar, troque só os nomes/tipos dos campos, o resto do pipeline
(PydanticOutputParser em chain.py) não precisa mudar.
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from typing import Literal


class AnaliseConsulta(BaseModel):
    """
    Estrutura de saída para quando o usuário pede uma análise sobre um
    ativo/tema. Usada pelo PydanticOutputParser no pipeline LCEL.
    """

    tema: str = Field(
        description="Assunto principal da consulta (ex: nome da empresa/ativo)."
    )
    categoria: Literal["fundamentalista", "tecnica", "macroeconomica", "geral"] = Field(
        description="Categoria da análise solicitada pelo usuário."
    )
    pontos_chave: list[str] = Field(
        description="Lista com os principais pontos levantados na resposta.",
        min_length=1,
    )
    nivel_confianca: float = Field(
        description="Confiança do modelo na resposta, de 0.0 a 1.0.",
        ge=0.0,
        le=1.0,
    )
    requer_disclaimer: bool = Field(
        default=True,
        description="Se True, a resposta deve incluir o aviso de que não é recomendação de investimento.",
    )

    @field_validator("tema")
    @classmethod
    def tema_nao_vazio(cls, valor):
        if not valor.strip():
            raise ValueError("O tema não pode ser vazio")
        return valor.strip()

    @field_validator("pontos_chave")
    @classmethod
    def pontos_nao_vazios(cls, valores):
        if any(not texto.strip() for texto in valores):
            raise ValueError("Pontos-chave não podem conter texto vazio")
        return [texto.strip() for texto in valores]


class DocumentoFonte(BaseModel):
    """Proveniência validada antes de entrar no banco vetorial."""
    model_config = ConfigDict(str_strip_whitespace=True)
    arquivo: str = Field(min_length=1)
    fonte: str = Field(min_length=1)
    titulo: str = Field(min_length=1)
    pagina: int = Field(ge=1)
    tipo: str = Field(min_length=1)
    categoria: str = Field(min_length=1)
    data: str = Field(min_length=1)

    @field_validator("fonte")
    @classmethod
    def origem_publica(cls, valor):
        from urllib.parse import urlparse
        url = urlparse(valor)
        if url.scheme not in ("http", "https") or not url.netloc:
            raise ValueError("Informe a URL HTTP(S) da publicação original")
        return valor


class ChunkFonte(DocumentoFonte):
    chunk_id: str = Field(min_length=1)


class RespostaRAG(BaseModel):
    """Contrato público com contextos e fontes alinhados para auditoria/RAGAS."""
    resposta: str = Field(min_length=1)
    fontes: list[ChunkFonte]
    contextos: list[str]

    @model_validator(mode="after")
    def fontes_alinhadas(self):
        if len(self.fontes) != len(self.contextos):
            raise ValueError("Cada contexto deve ter uma fonte correspondente")
        return self
