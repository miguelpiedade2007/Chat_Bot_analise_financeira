"""Verifica as evidências obrigatórias antes de montar o pacote final."""
import json
import hashlib
import re
from pathlib import Path
import nbformat
import pandas as pd
from app.config import ROOT, RESULTADOS, DOCUMENTOS, CHUNK_SIZES
from app.documentos import carregar_documentos


def main():
    docs = carregar_documentos()
    perguntas = json.loads((ROOT / "data/perguntas_teste.json").read_text(encoding="utf-8"))
    tabela = pd.read_csv(RESULTADOS / "ragas_comparacao.csv")
    for tamanho in CHUNK_SIZES:
        assert tabela.loc[tabela.chunk_size == tamanho, "user_input"].tolist() == perguntas
    assert len(tabela) == len(perguntas) * len(CHUNK_SIZES)
    for metrica in ("faithfulness", "answer_relevancy"):
        assert tabela[metrica].between(0, 1).all(), f"Métrica inválida: {metrica}"
    vencedor = json.loads((RESULTADOS / "configuracao_vencedora.json").read_text(encoding="utf-8"))
    assert vencedor["faithfulness"] >= .7, "Meta mínima de faithfulness não atingida"
    notebook = nbformat.read(ROOT / "notebooks/CKP02_DocMind_RAG.ipynb", as_version=4)
    nbformat.validate(notebook)
    for celula in notebook.cells:
        if celula.cell_type == "code":
            assert celula.execution_count is not None, "Notebook com célula não executada"
            assert not any(o.output_type == "error" for o in celula.outputs), "Notebook com erro"
    evidencias = json.loads((RESULTADOS / "evidencias_respostas.json").read_text(encoding="utf-8"))["dados"]
    assert len(evidencias) == len(tabela)
    for item in evidencias:
        validos = {f["chunk_id"] for f in item["fontes"]}
        citados = {c for grupo in re.findall(r"\[([^\]]+)\]", item["resposta"])
                   for c in re.findall(r"\b\d+_\d+\b", grupo)}
        assert not citados - validos, "Citação aponta para chunk não recuperado"
    resumo = {"documentos": len({d.metadata['arquivo'] for d in docs}),
              "paginas_textuais": len(docs), "linhas_ragas": len(tabela),
              "notebook_sem_erros": True, "vencedor": vencedor,
              "sha256_documentos": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(DOCUMENTOS.glob('*.pdf'))}}
    (RESULTADOS / "verificacao_entrega.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
