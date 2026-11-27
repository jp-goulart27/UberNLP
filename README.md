# Auditoria Automatizada de Despesas de Mobilidade Urbana com NLP e Geolocalização

Pipeline híbrido em Python desenvolvido para automatizar a auditoria e a triagem de conformidade de corridas corporativas em aplicativos de transporte, combinando **análise temporal**, **geolocalização com raios diferenciados** e **Processamento de Linguagem Natural (NLP) baseado em *embeddings* semânticos multilíngues**.

---

## Visão Geral da Arquitetura

O sistema resolve o problema da ausência de bases históricas previamente rotuladas cruzando três dimensões independentes de evidência:

1. **Indicadores Temporais (`main.py`):** Identifica deslocamentos realizados fora da janela comercial padrão (madrugada entre `00h00` e `05h59`, fins de semana e feriados nacionais via biblioteca `holidays`).
2. **Normalização Espacial e Geofencing Diferenciado (`classify.py`):**
   - Padroniza endereços brutos de GPS e agrupa variações de numeração predial próximas ($\pm 50$ números).
   - **Escritórios da Empresa (`100 m`):** Calcula a distância geodésica (Haversine) até as sedes cadastradas em `escritorios.csv` antes de consultar estabelecimentos comerciais.
   - **Hotéis e Aeroportos (`50 m`):** Identifica hubs de viagem via regras determinísticas e *Google Places API (Nearby Search)*.
   - **Restaurantes (`20 m`):** Utiliza raio restrito para evitar falsos positivos em regiões comerciais densas.
3. **NLP Semântico Multilíngue e Regras de Política (`main.py`):**
   - Utiliza o modelo neural pré-treinado `paraphrase-multilingual-MiniLM-L12-v2` (*Sentence Transformers*) para classificar justificativas em **Português, Espanhol e Inglês** nas categorias `Negócios`, `Viagem_Logistica`, `Generica` ou `Sem Justificativa` (limiar de similaridade de cosseno $\tau = 0{,}42$).
   - Aplica validações contextuais alinhadas à política interna (deslocamentos vinculados a escritórios corporativos e isenção para jantares/refeições em restaurantes).

---

## Resultados Experimentais

Avaliado sobre uma base operacional de **1.341 registros** (**1.310 corridas válidas** do tipo `Fare`), o pipeline apresentou os seguintes indicadores:

### Evolução do Módulo de NLP (TF-IDF vs. Embeddings Multilíngues)

| Categoria NLP | Linha de Base (TF-IDF) | Arquitetura Final (Embeddings + Regras) | Variação |
| :--- | :---: | :---: | :---: |
| `Viagem_Logistica` | 313 (23,9%) | **589 (45,0%)** | +88,2% |
| `Negócios` | 304 (23,2%) | **387 (29,5%)** | +27,3% |
| `Generica` (Alertas de texto vago) | 683 (52,1%) | **325 (24,8%)** | **-52,4%** |
| `Sem Justificativa` | 10 (0,8%) | **9 (0,7%)** | -10,0% |

### Estratificação da Fila de Prioridade (`score_suspeita`)

| Nível de Prioridade | Critério de Score | Corridas | % da Base | Ação Recomendada |
| :--- | :--- | :---: | :---: | :--- |
| **Prioridade Alta** | $\text{Score} \ge 3{,}0$ | **43** | **3,3%** | Revisão manual crítica imediata |
| **Prioridade Média** | $1{,}5 \le \text{Score} < 3{,}0$ | **121** | **9,2%** | Verificação secundária / amostragem |
| **Normal / Baixa** | $\text{Score} < 1{,}5$ | **1.146** | **87,5%** | Aprovação automática (conformidade) |

* **Desempenho computacional:** Processamento completo de ponta a ponta em **10,73 segundos** em CPU (após o cache inicial do modelo neural), vetorizando 888 descrições únicas em lote.

---

## Estrutura do Projeto

```text
auditoria-uber-nlp/
├── README.md                     # Documentação geral e resumo dos resultados
├── requirements.txt              # Dependências Python do projeto
├── .gitignore                    # Exclusão de dados sensíveis e arquivos temporários
├── main.py                       # Pipeline principal (Datas, NLP Embeddings, Regras e Score)
├── classify.py                   # Classificador geográfico híbrido (Haversine + Google Places API)
├── docs/
│   └── paper_auditoria_uber.pdf  # Artigo técnico compilado em LaTeX
└── dados/
    ├── escritorios_exemplo.csv   # Modelo de estrutura para endereços corporativos
    └── amostra_anonimizada.csv   # Amostra fictícia/anonimizada para testes reproduzíveis
```

---

## Como Executar Localmente

### 1. Clonar o repositório e criar o ambiente virtual

```bash
git clone https://github.com/SEU-USUARIO/auditoria-uber-nlp.git
cd auditoria-uber-nlp

python -m venv venv
# No Windows:
venv\Scripts\activate
# No Linux/macOS:
source venv/bin/activate
```

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 3. Configurar a chave da API do Google Maps (Opcional para classificação geográfica)

Defina a variável de ambiente antes de rodar o `classify.py`:

```bash
# Windows (PowerShell):
$env:GOOGLE_MAPS_API_KEY="SUA_CHAVE_AQUI"

# Linux/macOS:
export GOOGLE_MAPS_API_KEY="SUA_CHAVE_AQUI"
```

### 4. Executar o pipeline

```bash
# Etapa 1: Gera a lista consolidada de locais e calcula a triagem inicial
python main.py

# Etapa 2: Classifica os locais únicos (Escritório, Hotel, Aeroporto, Restaurante, Outros)
python classify.py

# Etapa 3: Reexecuta o main.py para cruzar as categorias geográficas e gerar a fila final
python main.py
```

Os resultados ranqueados serão exportados para `dados/fila_prioridade_auditoria.csv` e `dados/fila_prioridade_auditoria.xlsx`.
