# AnalistaBot — CKP02 DocMind RAG

**FIAP · Prompt Engineering and Artificial Intelligence · 2º semestre de 2026 · Turma 1CCPI**

**Integrantes**

- Miguel Piedade — RM 572445

## 1. Apresentação

O **AnalistaBot** é um assistente educacional sobre **análise de ações e mercado de capitais brasileiro**, desenvolvido a partir do domínio adotado no CKP01. Nesta segunda etapa, a aplicação incorpora *Retrieval-Augmented Generation* (RAG) para fundamentar respostas em documentos oficiais.

O projeto é voltado a estudantes e investidores iniciantes interessados em compreender instrumentos financeiros, riscos, indicadores e mecanismos de proteção ao investidor. O assistente não fornece cotações em tempo real, recomendações personalizadas de investimento ou garantias de rentabilidade.

O pipeline foi organizado em módulos reutilizáveis, incluindo a função `app.rag.buscar(consulta)`, prevista para integração como ferramenta na próxima etapa (CKP03).

## 2. Base de conhecimento

A base é composta por **cinco publicações da Comissão de Valores Mobiliários (CVM)**, preservadas em `data/documentos/`:

| Documento | Tema |
|---|---|
| Day Trade | Operações de curtíssimo prazo e seus riscos |
| BDR | Brazilian Depositary Receipts |
| MRP | Mecanismo de Ressarcimento de Prejuízos |
| Suitability | Adequação de produtos ao perfil do investidor |
| Investidor 60+ | Educação e proteção do investidor idoso |

Os arquivos de origem, títulos, categorias e respectivos **links oficiais** são identificados em `data/fontes.json` e no notebook. A base utilizada possui **160 páginas físicas**, das quais **152 contêm texto**, conforme o levantamento registrado na execução do projeto.



## 3. Arquitetura RAG

O processamento segue as etapas:

```text
Documentos PDF/TXT/MD
        ↓ load
Extração de texto e metadados
        ↓ split
RecursiveCharacterTextSplitter
        ↓ embed
nomic-embed-text (Ollama)
        ↓ store
ChromaDB persistente (local)
        ↓ retrieve
Busca semântica + filtro opcional por metadados
        ↓ generate
Gemma 4 via Ollama Cloud (temperature=0)
        ↓
Resposta fundamentada com citações de fontes
```

A implementação está distribuída principalmente entre `app/documentos.py` (ingestão e preparação) e `app/rag.py` (indexação, recuperação e geração). O `RecursiveCharacterTextSplitter` utiliza `separators=["\n\n", "\n", ". ", " ", ""]`.

Para o armazenamento vetorial é utilizado `chromadb.PersistentClient`, com coleções separadas por configuração: `analise_de_acoes_chunk_256` e `analise_de_acoes_chunk_512`. Os embeddings são produzidos com `OllamaEmbeddings(model="nomic-embed-text")`, incluindo operações de consulta e de documentos (`embed_query` e `embed_documents`). O modelo produz vetores de 768 dimensões.

**Infraestrutura de modelos:** o modelo gerador é `gemma4:cloud`, configurado com temperatura zero. Os embeddings utilizam exatamente o modelo `nomic-embed-text`, porém **são executados no Ollama local ou no ambiente do notebook**, enquanto a geração utiliza a nuvem.

## 4. Comparação de chunking e avaliação RAGAS

Foram comparadas duas configurações de divisão de texto, ambas com sobreposição de **12,5%**:

| Configuração | `chunk_size` | `chunk_overlap` | Total de chunks |
|---|---:|---:|---:|
| A | 256 | 32 | 835 |
| **B — escolhida** | **512** | **64** | **448** |

As duas configurações foram avaliadas com as **mesmas seis perguntas**. O RAGAS foi utilizado para medir *faithfulness* (fundamentação da resposta nos contextos recuperados) e *answer_relevancy* (relevância da resposta em relação à pergunta).

### Resultados experimentais

| Configuração | Faithfulness médio | Answer relevancy médio |
|---|---:|---:|
| 256/32 | 0,8958 | 0,8486 |
| **512/64** | **0,9237** | **0,8964** |

**Configuração escolhida: 512/64.** Ela apresentou maior *faithfulness* médio e maior *answer relevancy* médio no conjunto de seis perguntas. Os valores acima correspondem aos resultados informados para a execução de **09/10/2026**; a reprodução pode depender da disponibilidade dos modelos e serviços.

A seleção é feita pelo maior *faithfulness* médio, com *answer relevancy* como critério de desempate. A comparação principal foi realizada **sem reranking**, para isolar o efeito do tamanho dos chunks.

Os resultados detalhados e as evidências estão em:

- `resultados/ragas_comparacao.csv` — métricas por pergunta e configuração (12 avaliações);
- `resultados/ragas_medias.csv` — médias por configuração;
- `resultados/resumo_avaliacao.md` — síntese do experimento;
- `resultados/evidencias_respostas.json` — respostas, contextos e fontes utilizadas;
- `resultados/configuracao_vencedora.json` — configuração selecionada.

## 5. Funcionalidades adicionais

- **Filtragem por metadados:** documentos identificados por categoria e outros atributos, com filtros aplicados à recuperação no ChromaDB por meio de `where`.
- **Reranking opcional:** reordenação dos trechos recuperados com o cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`. Não altera a comparação principal de chunking.
- **Interface Gradio:** chat integrado ao RAG, apresentação de fontes e funcionalidades mantidas do CKP01.
- **Saídas estruturadas:** schemas Pydantic para validação de dados e proveniência.
- **Memória conversacional:** histórico por sessão com limite aproximado de 1.200 tokens na aba conversacional. Na aba RAG, as consultas devem ser autossuficientes, pois o contexto é recuperado dos documentos.

## 6. Requisitos e execução local

**Requisitos:** Python 3.11 ou 3.12, Ollama instalado, acesso à internet para as chamadas ao modelo na nuvem e uma chave `OLLAMA_API_KEY` válida. A instalação das dependências utiliza `requirements.txt`.

Na raiz do projeto, execute no PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure `OLLAMA_API_KEY` no `.env` (sem publicar o arquivo) e instale o modelo local de embeddings:

```powershell
ollama pull nomic-embed-text
```

Em seguida:

```powershell
python -m scripts.baixar_fontes
python -m scripts.indexar
python -m scripts.avaliar
python -m app.main
```

A interface é disponibilizada em **http://127.0.0.1:7860**. Os PDFs já incluídos em `data/documentos/` podem ser reutilizados, sem download adicional. Para executar testes automatizados:

```powershell
python -m pytest -q
```

O reranking é opcional e requer instalação adicional:

```powershell
pip install -r requirements-rerank.txt
```

As dependências foram fixadas para preservar a compatibilidade do ambiente de testes. Consulte `VERSOES_TESTADAS.txt` para o registro das versões.

## 7. Notebook Google Colab

**[Acessar o notebook no Google Colab](https://colab.research.google.com/drive/16VuxyZ66ZvLTrOr8QRwVJs4oU_It2T07?usp=sharing)**

Para reproduzir a atividade:

1. Abra `notebooks/CKP02_DocMind_RAG.ipynb` no Google Colab.
2. Cadastre `OLLAMA_API_KEY` nos **Secrets** do Colab e autorize o acesso ao notebook.
3. Execute todas as células, fornecendo o arquivo de entrega do projeto quando solicitado pela primeira célula.
4. Verifique a indexação, as seis perguntas, as métricas RAGAS e a tabela comparativa.
5. Salve o notebook **com os outputs**, habilite permissão de **leitor** pelo link e exporte uma cópia `.ipynb`.



## 8. Como adicionar documentos à base

1. Adicione um arquivo PDF textual, TXT ou Markdown (UTF-8) a `data/documentos/`.
2. Registre a origem em `data/fontes.json`, preenchendo `arquivo`, `titulo`, `url`, `tipo`, `data`, `categoria` e `instituicao`.
3. Execute `python -m scripts.indexar` para atualizar o índice vetorial.
4. Se a base de conhecimento mudar, revise as perguntas de teste e execute novamente `python -m scripts.avaliar` para atualizar as métricas e a escolha do chunking.

PDFs exclusivamente escaneados precisam passar por OCR antes da ingestão. Os originais são preservados.

## 9. Organização da entrega

A entrega inclui o código-fonte, o README, o notebook `.ipynb` executado, os cinco documentos reais da base e os arquivos de evidência da avaliação.



