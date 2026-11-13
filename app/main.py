"""Interface Gradio do CKP02: RAG com citações + CKP01 estruturado."""
import os
import gradio as gr
from app.chain import executar_interacao, build_llm
from app.rag import responder
from app.prompts import SYSTEM_PROMPT
from app.config import tamanho_selecionado
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import tiktoken

CATEGORIAS = {
    "Todos": None,
    "Negociação": "negociacao",
    "BDR": "bdr",
    "Proteção": "protecao",
    "Perfil do investidor": "perfil",
}


def responder_rag(mensagem, historico, filtro="Todos", usar_rerank=False):
    try:
        tamanho = tamanho_selecionado()
        selecao = {"categoria": CATEGORIAS[filtro]} if CATEGORIAS[filtro] else None
        resultado = responder(mensagem, chunk_size=tamanho, filtros=selecao, usar_rerank=usar_rerank)
        linhas = []
        ja_vistos = set()
        for f in resultado["fontes"]:
            identificador = (f["chunk_id"], f["arquivo"])
            if identificador in ja_vistos:
                continue
            ja_vistos.add(identificador)
            linhas.append(f"- [{f['chunk_id']}] **{f['titulo']}**, p. {f['pagina']} — {f['fonte']}")
        return resultado["resposta"] + ("\n\n**Trechos encontrados na base:**\n" + "\n".join(linhas) if linhas else "")
    except Exception as exc:
        return f"Erro ao pesquisar a base: {exc}. Confira download, indexação, Ollama e .env."


def responder_ckp01(mensagem, historico):
    """Chat do CKP01 com histórico isolado por sessão e teto aproximado de 1.200 tokens."""
    try:
        encoder = tiktoken.get_encoding("cl100k_base")
        disponiveis = []
        usados = 0
        # ChatInterface envia histórico no formato {role, content} no Gradio 5/6.
        for item in reversed(historico or []):
            if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
                continue
            conteudo = item.get("content", "")
            if not isinstance(conteudo, str):
                continue
            estimativa = len(encoder.encode(conteudo))
            if usados + estimativa > 1200:
                break
            usados += estimativa
            tipo = HumanMessage if item["role"] == "user" else AIMessage
            disponiveis.append(tipo(content=conteudo))
        mensagens = [SystemMessage(content=SYSTEM_PROMPT), *reversed(disponiveis),
                     HumanMessage(content=mensagem)]
        return str(build_llm().invoke(mensagens).content)
    except Exception as exc:
        return f"Erro no chat conversacional: {exc}"


def responder_estruturado(pergunta):
    try:
        return executar_interacao(pergunta).model_dump_json(indent=2)
    except Exception as exc:
        return f"Erro: {exc}"


def construir_interface():
    with gr.Blocks(title="AnalistaBot - CKP02 DocMind RAG") as demo:
        gr.Markdown("# AnalistaBot · CKP02 — DocMind RAG")
        gr.Markdown("Respostas documentais sobre análise de ações; fonte e página em cada busca. Uso educacional.")
        with gr.Tab("Perguntar à base (RAG)"):
            with gr.Row():
                filtro = gr.Dropdown(choices=list(CATEGORIAS), value="Todos", label="Filtro por categoria")
                rerank = gr.Checkbox(value=os.getenv("RAG_RERANK", "0") == "1", label="Reranking cross-encoder")
            gr.ChatInterface(
                fn=responder_rag, additional_inputs=[filtro, rerank],
                examples=[["O que é um BDR?", "Todos", False],
                          ["Explique a diferença entre day trade e swing trade", "Todos", False],
                          ["Quais cuidados são indicados aos investidores 60+?", "Todos", False]],
            )
        with gr.Tab("CKP01 · Chat conversacional"):
            gr.ChatInterface(fn=responder_ckp01, examples=["O que significa P/L?", "Qual a diferença entre ROE e ROA?"])
        with gr.Tab("CKP01 · Saída estruturada"):
            texto = gr.Textbox(label="Pergunta educativa", lines=2)
            saida = gr.Code(label="JSON Pydantic", language="json")
            gr.Button("Gerar análise").click(responder_estruturado, inputs=texto, outputs=saida)
        gr.Markdown("*As fontes são educacionais e podem citar regras da época da publicação; consulte normas vigentes.*")
    return demo


if __name__ == "__main__":
    construir_interface().launch(server_name="127.0.0.1", server_port=7860, share=False)
