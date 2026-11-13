"""
Gerenciamento de memória da conversa.

Estratégia escolhida: TokenBuffer.

Justificativa: o requisito já pede um limite de 800–1500 tokens, o que
bate exatamente com o funcionamento do TokenBuffer. Em análise de ações,
preservar o texto exato dos últimos indicadores/tickers discutidos é
mais importante do que manter um resumo compactado (que poderia perder
precisão numérica), e o TokenBuffer garante um teto de custo previsível
por chamada sem depender de uma chamada extra ao LLM para resumir.
"""

# A partir do langchain 1.x, as classes de memória "legadas" também
# foram movidas para o pacote separado langchain-classic.
from langchain_classic.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationTokenBufferMemory,
)
from langchain_ollama import ChatOllama


def get_memory(llm: ChatOllama, estrategia: str = "summary"):
    """
    Retorna o objeto de memória configurado conforme a estratégia escolhida.

    estrategia: "buffer" | "summary" | "token_buffer"
    """
    if estrategia == "buffer":
        # Guarda o histórico completo da conversa. Simples, mas cresce
        # rápido em tokens — bom pra conversas curtas.
        return ConversationBufferMemory(memory_key="history", return_messages=True)

    if estrategia == "summary":
        # Resume a conversa continuamente usando o próprio LLM. Bom
        # equilíbrio entre custo de tokens e manter contexto relevante
        # em conversas longas.
        return ConversationSummaryMemory(
            llm=llm, memory_key="history", return_messages=True
        )

    if estrategia == "token_buffer":
        # Mantém um buffer limitado por número de tokens (800–1500),
        # descartando as trocas mais antigas quando o limite é atingido.
        return ConversationTokenBufferMemory(
            llm=llm,
            memory_key="history",
            return_messages=True,
            max_token_limit=1200,
        )

    raise ValueError(f"Estratégia de memória desconhecida: {estrategia}")
