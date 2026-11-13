import gradio as gr
from app.chain import executar_interacao, conversation_chain


def responder_chat(mensagem: str, historico: list) -> str:
    """Responde no modo chat contínuo, utilizando a memória TokenBuffer."""
    try:
        return conversation_chain.predict(input=mensagem)
    except Exception as e:
        return f"Erro na comunicação com o LLM: {str(e)}"


def limpar_memoria():
    """Esvazia a memória interna do LangChain e limpa a janela do chat."""
    conversation_chain.memory.clear()
    return []  # Retorna lista vazia para resetar o componente gr.Chatbot


def gerar_analise(pergunta: str) -> str:
    """Executa a arquitetura de 2 chains para retornar o JSON validado pelo Pydantic."""
    try:
        resultado_estruturado = executar_interacao(pergunta)
        return resultado_estruturado.model_dump_json(indent=2)
    except Exception as e:
        return f"Erro na formatação Pydantic: {str(e)}"


with gr.Blocks(title="AnalistaBot - CKP01") as demo:
    gr.Markdown("# AnalistaBot — Chatbot de Análise de Ações (CKP01)")
    gr.Markdown("Assistente para investidores iniciantes com memória TokenBuffer e validação Pydantic v2.")

    with gr.Tab("Chat Conversacional"):
        chatbot_widget = gr.Chatbot(height=400)

        # Interface de chat usando os componentes desacoplados
        chat = gr.ChatInterface(
            fn=responder_chat,
            chatbot=chatbot_widget,
            textbox=gr.Textbox(
                placeholder="Pergunte sobre um indicador ou empresa...",
                container=False,
                scale=7
            )
        )

        # Botão explícito para reiniciar o contexto e a UI
        btn_reiniciar = gr.Button("Reiniciar Conversa", variant="secondary")
        btn_reiniciar.click(fn=limpar_memoria, inputs=None, outputs=chatbot_widget)

    with gr.Tab("Análise Estruturada (Pydantic)"):
        entrada = gr.Textbox(label="Pergunta para análise estruturada (ex: Qual a análise fundamentalista de PETR4?)")
        botao = gr.Button("Gerar análise JSON", variant="primary")
        saida = gr.Code(label="Saída Validada", language="json", lines=15)

        botao.click(fn=gerar_analise, inputs=entrada, outputs=saida)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Default())