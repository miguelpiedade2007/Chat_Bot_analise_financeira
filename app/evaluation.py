"""Comparação quantitativa com métricas reais RAGAS 0.3.x."""
import json
import math
from datetime import datetime, timezone
import pandas as pd
from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from app.chain import build_llm
from app.config import CHUNK_SIZES, RESULTADOS, ROOT
from app.rag import responder, build_embeddings


def perguntas_avaliacao():
    perguntas = json.loads((ROOT / "data" / "perguntas_teste.json").read_text(encoding="utf-8"))
    if len(perguntas) < 5:
        raise ValueError("A avaliação exige pelo menos cinco perguntas")
    return perguntas


def avaliar_configuracoes(chunk_sizes=CHUNK_SIZES, perguntas=None):
    """Gera métricas sem inventar notas; salva respostas/contextos auditáveis."""
    perguntas = perguntas or perguntas_avaliacao()
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    # Resultados anteriores não podem continuar parecendo aprovados após falha.
    for nome in ("configuracao_vencedora.json", "ragas_medias.csv", "resumo_avaliacao.md"):
        (RESULTADOS / nome).unlink(missing_ok=True)
    llm = build_llm()
    embeds = build_embeddings()
    conjunto = []
    tabelas = []
    for tamanho in chunk_sizes:
        amostras = []
        for pergunta in perguntas:
            resultado = responder(pergunta, chunk_size=tamanho, llm=llm, embeddings=embeds)
            amostras.append({"user_input": pergunta, "response": resultado["resposta"],
                            "retrieved_contexts": resultado["contextos"]})
            conjunto.append({"chunk_size": tamanho, "pergunta": pergunta,
                             "resposta": resultado["resposta"], "fontes": resultado["fontes"],
                             "contextos": resultado["contextos"]})
        dataset = EvaluationDataset.from_list(amostras)
        pontuacao = evaluate(
            dataset=dataset,
            metrics=[Faithfulness(), ResponseRelevancy()],
            llm=LangchainLLMWrapper(llm),
            embeddings=LangchainEmbeddingsWrapper(embeds),
            raise_exceptions=False,
            show_progress=True,
            run_config=RunConfig(timeout=240, max_retries=2, max_workers=1),
        )
        df = pontuacao.to_pandas()
        # Permanece explícito quando o RAGAS não consegue medir uma resposta.
        resultado_df = df[["user_input", "response", "faithfulness", "answer_relevancy"]].copy()
        resultado_df.insert(0, "chunk_size", tamanho)
        resultado_df.to_csv(RESULTADOS / f"ragas_chunk_{tamanho}.csv", index=False, encoding="utf-8")
        tabelas.append(resultado_df)
        # Preserva a evidência já obtida mesmo que a próxima configuração falhe.
        (RESULTADOS / "evidencias_respostas.json").write_text(
            json.dumps({"modelo_chat": llm.model, "modelo_embedding": "nomic-embed-text",
                        "dados": conjunto}, ensure_ascii=False, indent=2), encoding="utf-8")
    tabela = pd.concat(tabelas, ignore_index=True)
    tabela.to_csv(RESULTADOS / "ragas_comparacao.csv", index=False)
    (RESULTADOS / "evidencias_respostas.json").write_text(
        json.dumps({"criado_utc": datetime.now(timezone.utc).isoformat(),
                    "modelo_chat": llm.model, "modelo_embedding": "nomic-embed-text",
                    "dados": conjunto}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    invalidos = tabela[["faithfulness", "answer_relevancy"]].isna()
    if invalidos.any().any():
        raise RuntimeError(
            "RAGAS retornou métrica ausente/NaN. Confira os CSVs, o LLM juiz e os prompts; "
            "não considere a comparação validada até corrigir."
        )
    media = (tabela.groupby("chunk_size", as_index=False)[["faithfulness", "answer_relevancy"]]
             .mean().sort_values(["faithfulness", "answer_relevancy"], ascending=False))
    if not all(math.isfinite(float(x)) for x in media[["faithfulness", "answer_relevancy"]].to_numpy().flat):
        raise RuntimeError("Métricas não finitas: impossível justificar vencedora")
    vencedor = int(media.iloc[0]["chunk_size"])
    media.to_csv(RESULTADOS / "ragas_medias.csv", index=False)
    comentario = (
        f"# Comparação RAGAS medida\n\n"
        f"- Melhor chunk_size (critério: faithfulness médio; desempate por answer_relevancy): **{vencedor}**.\n"
        f"- Média de faithfulness: **{media.iloc[0]['faithfulness']:.3f}**; "
        f"answer_relevancy: **{media.iloc[0]['answer_relevancy']:.3f}**.\n"
        f"- Meta >=0,7: **{'atingida' if media.iloc[0]['faithfulness'] >= .7 else 'não atingida'}**.\n"
        f"- Cautela: resultados medidos pelo RAGAS, não garantias de factualidade.\n"
        f"- Faça revisão humana das respostas e dos trechos citados antes da entrega.\n"
    )
    (RESULTADOS / "resumo_avaliacao.md").write_text(comentario, encoding="utf-8")
    (RESULTADOS / "configuracao_vencedora.json").write_text(
        json.dumps({"chunk_size": vencedor, "faithfulness": float(media.iloc[0]['faithfulness']),
                    "answer_relevancy": float(media.iloc[0]['answer_relevancy'])}, indent=2), encoding="utf-8")
    return tabela, media, vencedor
