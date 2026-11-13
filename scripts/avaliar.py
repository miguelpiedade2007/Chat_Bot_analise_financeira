"""Avaliação RAGAS da mesma lista de perguntas nos dois chunkings."""
from app.evaluation import avaliar_configuracoes


def main():
    tabela, medias, vencedor = avaliar_configuracoes()
    print("\nRESULTADOS POR PERGUNTA:")
    print(tabela[["chunk_size", "user_input", "faithfulness", "answer_relevancy"]].to_string(index=False))
    print("\nMÉDIAS POR CONFIGURAÇÃO:\n", medias.to_string(index=False))
    print(f"\nCHUNK_SIZE ESCOLHIDO PELOS DADOS: {vencedor}")
    print(f"Altere RAG_CHUNK_SIZE={vencedor} no seu .env antes de usar o Gradio.")


if __name__ == "__main__":
    main()
