import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_classic.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser

# Importações dos seus módulos locais corrigidas
from app.memory_manager import get_memory
from app.schemas import AnaliseConsulta
from app.prompts import SYSTEM_PROMPT

load_dotenv()


def build_llm():
    """Retorna a instância do modelo para ser reaproveitada no sistema e no context_rot."""
    return ChatOllama(
        model="gemma4:cloud",
        api_key=os.getenv("OLLAMA_API_KEY"),
        temperature=0.2
    )


llm = build_llm()

# APLICAÇÃO DO TOKEN BUFFER
# Chama a função do seu memory_manager definindo a estratégia correta
memory = get_memory(llm=llm, estrategia="token_buffer")

# Chain 1: ConversationChain (Gestão de Memória e Chat)
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    # A variável 'history' deve ser idêntica ao memory_key do memory_manager.py
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

conversation_chain = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=chat_prompt,
    verbose=False
)

# Chain 2: Pipeline LCEL para Saída Estruturada (Pydantic)
parser = PydanticOutputParser(pydantic_object=AnaliseConsulta)

estruturacao_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Extraia as informações da resposta do analista e formate no esquema JSON exigido.\n{format_instructions}"),
    ("human", "{resposta_bruta}")
])

# Operador Pipe (|) exigido nos requisitos
lcel_structured_chain = estruturacao_prompt | llm | parser


def executar_interacao(user_input: str) -> AnaliseConsulta:
    """Função principal que orquestra as duas chains."""
    resposta_bruta = conversation_chain.predict(input=user_input)

    saida_estruturada = lcel_structured_chain.invoke({
        "resposta_bruta": resposta_bruta,
        "format_instructions": parser.get_format_instructions()
    })

    return saida_estruturada