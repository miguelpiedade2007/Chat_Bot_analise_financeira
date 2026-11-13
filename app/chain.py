"""Ollama Cloud + chains LCEL do CKP01; instanciadas sob demanda."""
import os
from functools import lru_cache
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from app.config import CHAT_MODEL
from app.memory_manager import get_memory
from app.prompts import SYSTEM_PROMPT
from app.schemas import AnaliseConsulta


@lru_cache(maxsize=1)
def build_llm():
    """Somente gemma4:cloud, temperature=0; chave fica em variável ambiente."""
    key = os.getenv("OLLAMA_API_KEY", "").strip()
    base_url = os.getenv("OLLAMA_CHAT_BASE_URL", "https://ollama.com").rstrip("/")
    from urllib.parse import urlparse
    local = urlparse(base_url).hostname in ("localhost", "127.0.0.1", "::1")
    if not local and (not key or key == "sua_chave_aqui"):
        raise RuntimeError("Defina OLLAMA_API_KEY no .env (ou Secrets do Colab).")
    # A API direta usa o identificador remoto; o daemon local resolve :cloud.
    modelo = CHAT_MODEL if local else "gemma4:31b"
    return ChatOllama(
        model=modelo,
        base_url=base_url,
        client_kwargs={"headers": {"Authorization": f"Bearer {key}"} if key else {}, "timeout": 180},
        temperature=0,
    )


@lru_cache(maxsize=1)
def build_ckp01_chain():
    """Preserva a ideia do chatbot conversacional com memória TokenBuffer."""
    llm = build_llm()
    memoria = get_memory(llm=llm, estrategia="token_buffer")
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])
    # Uso explícito de LCEL, com memória externa gerenciada no invocador.
    return prompt | llm | StrOutputParser(), memoria


def executar_interacao(texto, sessao=None):
    """Gera saída estruturada do antigo CKP01 usando o pipeline LCEL."""
    llm = build_llm()
    parser = PydanticOutputParser(pydantic_object=AnaliseConsulta)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + "\n{format_instructions}"),
        ("human", "{input}"),
    ])
    return (prompt | llm | parser).invoke({
        "input": texto, "format_instructions": parser.get_format_instructions()
    })
