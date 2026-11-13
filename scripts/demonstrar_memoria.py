"""Registra cinco turnos reais para tratar a pendência de demonstração do CKP01."""
import json
from app.config import RESULTADOS
from app.main import responder_ckp01


def main():
    historico = []
    perguntas = [
        "Meu nome é Marina. Estou estudando BDRs, sem intenção de comprar agora. Guarde esse contexto.",
        "Minha prioridade neste estudo é compreender os riscos. Registre essa prioridade.",
        "Prefiro explicações curtas, sem jargões. Confirme minha preferência.",
        "Quero montar uma lista de dúvidas para levar ao professor. Qual o primeiro tema que mencionei?",
        "Recapitule meu nome, o instrumento que estou estudando, minha prioridade e minha preferência de explicação.",
    ]
    for pergunta in perguntas:
        resposta = responder_ckp01(pergunta, historico)
        if resposta.startswith("Erro no chat"):
            raise RuntimeError(resposta)
        historico.extend([{"role": "user", "content": pergunta},
                          {"role": "assistant", "content": resposta}])
        print(f"Turno {len(historico) // 2}: {resposta}", flush=True)
    RESULTADOS.mkdir(exist_ok=True)
    (RESULTADOS / "demonstracao_memoria.json").write_text(
        json.dumps(historico, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
