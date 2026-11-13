import random
import re
import tiktoken
from app.chain import build_llm
from langchain_core.prompts import ChatPromptTemplate
from app.prompts import SYSTEM_PROMPT

encoder = tiktoken.get_encoding("cl100k_base")
random.seed(42)

PERGUNTA_TESTE = (
    "Liste TODOS os tickers que mencionei nesta conversa junto com o "
    "P/L de cada um. Responda no formato 'TICKER: valor', um por linha, "
    "sem explicações adicionais."
)

JANELAS = [10, 20, 40, 80, 150]

# Bloco de distração para exaurir a atenção do modelo
TEXTO_RUIDO = """
A análise fundamentalista exige a avaliação de diversos indicadores macroeconômicos.
Durante o último trimestre, o mercado observou uma flutuação nas taxas de juros,
impactando diretamente a precificação de ativos de renda variável e fixa. 
Investidores iniciantes frequentemente confundem volatilidade com risco permanente, 
o que pode levar a decisões precipitadas em momentos de correção de mercado. 
A diversificação do portfólio entre setores não correlacionados continua sendo a 
principal estratégia de mitigação. A inflação subjacente demonstra resiliência,
obrigando o Banco Central a manter uma política monetária restritiva por mais tempo.
""" * 10  # Gera aproximadamente 600-800 tokens de ruído por inserção


def gerar_fatos(quantidade: int) -> dict[str, str]:
    return {
        f"TCK{i}11": str(round(random.uniform(2.0, 30.0), 2))
        for i in range(1, quantidade + 1)
    }


def montar_historico(fatos: dict[str, str]) -> str:
    turnos = []
    for ticker, pl in fatos.items():
        # Fato (Sinal)
        turnos.append(
            f"Usuário: o ticker {ticker} está com P/L de {pl}.\n"
            f"Assistente: anotado, {ticker} com P/L de {pl}.\n"
        )
        # Injeção de Ruído (Distrator)
        turnos.append(
            f"Usuário: Falando sobre macroeconomia...\n{TEXTO_RUIDO}\n"
            f"Assistente: Compreendo o cenário macroeconômico descrito.\n"
        )
    return "".join(turnos)


def contar_tokens(texto: str) -> int:
    return len(encoder.encode(texto))


def calcular_acerto(resposta: str, gabarito: dict[str, str]) -> float:
    encontrados = dict(re.findall(r"(TCK\d+11)\s*[:\-]\s*([\d.]+)", resposta))
    acertos = sum(
        1 for ticker, pl in gabarito.items() if encontrados.get(ticker) == pl
    )
    return round(100 * acertos / len(gabarito), 1)


def rodar_teste():
    llm = build_llm()
    resultados = []

    for n in JANELAS:
        gabarito = gerar_fatos(n)
        historico = montar_historico(gabarito)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{historico}\n{pergunta}"),
            ]
        )

        texto_prompt = prompt.format(historico=historico, pergunta=PERGUNTA_TESTE)
        tokens_tiktoken = contar_tokens(texto_prompt)

        chain = prompt | llm

        try:
            resposta = chain.invoke({"historico": historico, "pergunta": PERGUNTA_TESTE})
            texto_resposta = resposta.content.strip()
            metadata = resposta.response_metadata or {}
            tokens_ollama = metadata.get("prompt_eval_count", "N/A")
            pct_acerto = calcular_acerto(texto_resposta, gabarito)
        except Exception as e:
            texto_resposta = f"Erro: {e}"
            tokens_ollama = "Erro"
            pct_acerto = 0.0

        resultados.append(
            {
                "fatos": n,
                "tiktoken": tokens_tiktoken,
                "ollama": tokens_ollama,
                "pct_acerto": pct_acerto,
            }
        )

        # Parada de segurança para evitar estourar a RAM do servidor local acidentalmente
        if isinstance(tokens_ollama, int) and tokens_ollama > 60000:
            print("Limite seguro de tokens atingido para execução local.")
            break

    return resultados


def imprimir_tabela(resultados):
    print(f"{'Fatos':<7} | {'Tiktoken':<9} | {'Ollama':<8} | {'% Acerto':<9}")
    print("-" * 45)
    for r in resultados:
        print(f"{r['fatos']:<7} | {r['tiktoken']:<9} | {str(r['ollama']):<8} | {r['pct_acerto']:<9}")


if __name__ == "__main__":
    print("Testando recuperação multi-fato (context rot) com injeção de ruído...")
    resultados = rodar_teste()
    imprimir_tabela(resultados)