import os

# ==========================================
# 0. CONFIGURAÇÕES DE AMBIENTE (ANTI-TRAVAMENTO WINDOWS/PROXY)
# Deve permanecer no topo absoluto, antes de importar torch/transformers
# ==========================================
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"

import time
import holidays
import pandas as pd
import numpy as np
import re
import unicodedata
from collections import defaultdict

tempo_inicio_total = time.perf_counter()
tempos_etapas = {}

def registrar_etapa(nome_etapa, t_inicio):
    duracao = time.perf_counter() - t_inicio
    tempos_etapas[nome_etapa] = duracao
    print(f"   -> Concluído em {duracao:.2f}s\n", flush=True)

print("==================================================", flush=True)
print("INICIANDO PIPELINE DE AUDITORIA UBER (NLP + GEO)", flush=True)
print("==================================================\n", flush=True)

# ==========================================
# 1. CONFIGURAÇÃO E LEITURA DA BASE
# ==========================================
t_step = time.perf_counter()
print("[1/7] Lendo base de corridas...", flush=True)
os.makedirs("dados", exist_ok=True)

arquivo = "dados/2026.csv"
if not os.path.exists(arquivo):
    arquivo = "2026.csv"

try:
    df = pd.read_csv(arquivo, header=4, encoding="utf-8", sep=";")
    if len(df.columns) <= 2:
        df = pd.read_csv(arquivo, header=4, encoding="utf-8", sep=",")
except Exception:
    df = pd.read_csv(arquivo, sep=";", encoding="utf-8")

if len(df.columns) >= 13:
    df.columns = [
        "data", "hora", "nome", "sobrenome", "id_funcionario", "servico",
        "cidade", "origem", "destino", "tipo_transacao", "valor_local",
        "valor_brl", "descricao"
    ][:len(df.columns)]

print(f"   Arquivo '{arquivo}' carregado ({len(df)} registros brutos).", flush=True)
registrar_etapa("1. Leitura CSV", t_step)

# ==========================================
# 2. FILTRO E TRATAMENTO DE DATAS
# ==========================================
t_step = time.perf_counter()
print("[2/7] Filtrando corridas válidas e tratando datas/horários...", flush=True)
if "tipo_transacao" in df.columns:
    df = df[df["tipo_transacao"] == "Fare"].copy()
df.reset_index(drop=True, inplace=True)

df["data"] = pd.to_datetime(df["data"], format="%m/%d/%Y", errors="coerce")
df["data_hora"] = pd.to_datetime(
    df["data"].dt.strftime("%Y-%m-%d") + " " + df["hora"].astype(str),
    format="%Y-%m-%d %I:%M%p",
    errors="coerce"
)
df["hora_num"] = df["data_hora"].dt.hour
print(f"   Total de corridas analisáveis ('Fare'): {len(df)}.", flush=True)
registrar_etapa("2. Tratamento de Datas", t_step)

# ==========================================
# 3. ALERTAS TEMPORAIS
# ==========================================
t_step = time.perf_counter()
print("[3/7] Calculando alertas temporais (madrugada, fim de semana, feriados)...", flush=True)
df["fim_de_semana"] = df["data_hora"].dt.dayofweek >= 5
df["madrugada"] = df["hora_num"].between(0, 5)

feriados_br = holidays.Brazil(years=df["data"].dt.year.dropna().unique())
df["feriado"] = df["data"].dt.date.apply(lambda d: d in feriados_br if pd.notna(d) else False)

df["qtd_alertas_temporais"] = (
    df["madrugada"].astype(int)
    + df["fim_de_semana"].astype(int)
    + df["feriado"].astype(int)
)
print(f"   Corridas com pelo menos 1 alerta temporal: {(df['qtd_alertas_temporais'] > 0).sum()}.", flush=True)
registrar_etapa("3. Alertas Temporais", t_step)

# ==========================================
# 4. MÓDULO DE NLP COM EMBEDDINGS MULTILÍNGUES
# ==========================================
t_step = time.perf_counter()
print("[4/7] Importando bibliotecas de IA e carregando modelo de embeddings...", flush=True)
from sentence_transformers import SentenceTransformer, util

MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"
THRESHOLD_SIMILARIDADE = 0.42

try:
    modelo_nlp = SentenceTransformer(MODELO_EMBEDDINGS, local_files_only=True)
    print(f"   Modelo '{MODELO_EMBEDDINGS}' carregado direto do cache local.", flush=True)
except Exception:
    print(f"   Baixando modelo '{MODELO_EMBEDDINGS}' (primeira execução)...", flush=True)
    modelo_nlp = SentenceTransformer(MODELO_EMBEDDINGS)

registrar_etapa("4a. Carregamento Modelo IA", t_step)

t_step = time.perf_counter()
print("[4.5/7] Classificando justificativas com Embeddings Semânticos...", flush=True)

FRASES_ANCORA_NLP = {
    "Negócios": [
        "Reunião com cliente ou fornecedor",
        "Visita técnica ao cliente, obra, mina ou laboratório",
        "Deslocamento para o escritório ou sede da empresa",
        "Alinhamento de projeto, contrato ou proposta comercial",
        "Treinamento, workshop, congresso ou evento corporativo",
        "Jantar ou almoço corporativo com cliente ou equipe",
        "Entrega de material, documento, notebook ou serviço em banco e cartório",
        "Levar ou buscar carro da empresa na oficina",
        "Meeting with client, project alignment or business event",
        "Reunión de trabajo en la oficina o visita a cliente"
    ],
    "Viagem_Logistica": [
        "Deslocamento para o aeroporto para embarque ou desembarque de voo",
        "Translado entre aeroporto e hotel durante viagem a trabalho",
        "Retorno ou ida para o hotel ou hospedagem",
        "Deslocamento para rodoviária ou estação de trem",
        "Airport transfer, flight departure or arrival",
        "Back to hotel or trip logistics",
        "Traslado al aeropuerto o regreso al hotel"
    ],
    "Generica": [
        "Uber",
        "Corrida",
        "Deslocamento",
        "Ida ou volta para casa",
        "Trajeto ou transporte",
        "Retorno",
        "Going home or commute",
        "Regreso a casa"
    ]
}

ancora_textos = []
ancora_categorias = []
for cat, frases in FRASES_ANCORA_NLP.items():
    for frase in frases:
        ancora_textos.append(frase)
        ancora_categorias.append(cat)

embeddings_ancoras = modelo_nlp.encode(
    ancora_textos,
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=False
)

def preparar_texto_embedding(texto):
    if pd.isna(texto):
        return ""
    s = re.sub(r"\s+", " ", str(texto)).strip()
    s_alfanum = re.sub(r"[^\w\s]", "", s).strip()
    if len(s_alfanum) <= 2:
        return ""
    return s

def classificar_justificativas_otimizado(serie_descricoes):
    textos_limpos = [preparar_texto_embedding(t) for t in serie_descricoes]
    # Deduplica textos válidos para evitar calcular o mesmo embedding várias vezes
    textos_unicos = list(dict.fromkeys([t for t in textos_limpos if t != ""]))

    print(f"   Vetorizando {len(textos_unicos)} descrições únicas (de {len(textos_limpos)} corridas)...", flush=True)

    mapa_resultados = {"": ("Sem Justificativa", 0.0)}

    if textos_unicos:
        embeddings_unicos = modelo_nlp.encode(
            textos_unicos,
            batch_size=64,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        matriz_sim = util.cos_sim(embeddings_unicos, embeddings_ancoras).cpu().numpy()

        idxs_por_cat = {
            cat: [j for j, c in enumerate(ancora_categorias) if c == cat]
            for cat in FRASES_ANCORA_NLP.keys()
        }

        for idx, texto_u in enumerate(textos_unicos):
            sims_linha = matriz_sim[idx]
            melhor_por_cat = {
                cat: float(np.max(sims_linha[idxs]))
                for cat, idxs in idxs_por_cat.items()
            }

            if (
                melhor_por_cat["Viagem_Logistica"] >= THRESHOLD_SIMILARIDADE
                and melhor_por_cat["Viagem_Logistica"] >= melhor_por_cat["Negócios"]
            ):
                mapa_resultados[texto_u] = ("Viagem_Logistica", round(melhor_por_cat["Viagem_Logistica"], 3))
            elif (
                melhor_por_cat["Negócios"] >= THRESHOLD_SIMILARIDADE
                and melhor_por_cat["Negócios"] > melhor_por_cat["Viagem_Logistica"]
            ):
                mapa_resultados[texto_u] = ("Negócios", round(melhor_por_cat["Negócios"], 3))
            else:
                melhor_cat = max(melhor_por_cat, key=melhor_por_cat.get)
                maior_sim = melhor_por_cat[melhor_cat]
                cat_final = "Generica" if maior_sim < THRESHOLD_SIMILARIDADE else melhor_cat
                mapa_resultados[texto_u] = (cat_final, round(maior_sim, 3))

    categorias_finais = [mapa_resultados[t][0] for t in textos_limpos]
    scores_finais = [mapa_resultados[t][1] for t in textos_limpos]
    return categorias_finais, scores_finais

df["categoria_justificativa_nlp"], df["score_confianca_nlp"] = classificar_justificativas_otimizado(df["descricao"])
registrar_etapa("4b. Inferência Embeddings", t_step)

# ==========================================
# 5. PIPELINE DE LIMPEZA E AGRUPAMENTO DE ENDEREÇOS
# ==========================================
t_step = time.perf_counter()
print("[5/7] Normalizando e agrupando endereços de origem e destino...", flush=True)

def normalizar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto).lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^\w\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()

def pipeline_limpeza_completa(addr):
    if not isinstance(addr, str) or pd.isna(addr):
        return ""

    s = addr.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))

    s = re.sub(r",\s*(brasil|brazil|usa?|chile|colombia|arabia saudita|el salvador)\b.*$", "", s)
    s = re.sub(r"\b\d{5}\s+\d{4}\b", "", s)
    s = re.sub(r"\b[a-z]\d[a-z]\s*\d[a-z]\d\b", "", s)
    s = re.sub(r"\b\d{5}\s?-?\s?\d{3}\b", " ", s)
    s = re.sub(r"\b\d{5}\s*$", " ", s)

    abbrevs = {
        r"\br\.\s*": "rua ", r"\br\b\s+": "rua ",
        r"\bav\.\s*": "avenida ", r"\bav\b\s+": "avenida ",
        r"\brod\.\s*": "rodovia ", r"\brod\b\s+": "rodovia ",
        r"\bal\.\s*": "alameda ", r"\bal\b\s+": "alameda ",
        r"\btv\.\s*": "travessa ", r"\btv\b\s+": "travessa ",
        r"\bpc\.\s*": "praca ", r"\bpz\.\s*": "praca ", r"\bprc\.\s*": "praca ",
        r"\bst\.\s*": "setor ", r"\bst\b\s+": "setor ",
        r"\bqd\.\s*": "quadra ", r"\bqd\b\s+": "quadra ",
        r"\blt\.\s*": "lote ", r"\blt\b\s+": "lote ",
        r"\bcj\.\s*": "conjunto ", r"\bcj\b\s+": "conjunto ",
        r"\btr\.\s*": "trecho ", r"\btr\b\s+": "trecho ",
        r"\bcel\b": "coronel", r"\bdr\b": "doutor", r"\bprof\b": "professor",
        r"\bgov\b": "governador", r"\bpres\b": "presidente", r"\bsrg\b": "sargento"
    }
    for k, v in abbrevs.items():
        s = re.sub(k, v, s)

    if re.search(r"\b(confins|cnf)\b", s) or ("lmg 800" in s and "confins" in s):
        return "aeroporto internacional de confins cnf confins mg"
    elif re.search(r"\b(santa genoveva|gyn)\b", s) or ("aeroporto" in s and "goiania" in s and "alameda" in s):
        return "aeroporto internacional santa genoveva gyn goiania go"
    elif re.search(r"\b(guararapes|recife guararapes|rec)\b", s) and "recife" in s:
        return "aeroporto internacional dos guararapes rec recife pe"
    elif re.search(r"\b(presidente juscelino|bsb)\b", s) or ("aeroporto" in s and "brasilia" in s):
        return "aeroporto internacional de brasilia bsb brasilia df"
    elif re.search(r"\b(viracopos|vcp)\b", s):
        return "aeroporto internacional de viracopos vcp campinas sp"
    elif re.search(r"\b(congonhas|cgh)\b", s):
        return "aeroporto de congonhas cgh sao paulo sp"
    elif re.search(r"\b(cumbica|gru)\b", s) or ("guarulhos" in s and "airport" in s):
        return "aeroporto internacional de guarulhos gru guarulhos sp"
    elif re.search(r"\b(natal airport|nat)\b", s) and "natal" in s:
        return "aeroporto internacional de natal nat sao goncalo do amarante rn"
    elif re.search(r"\b(miami international airport|mia)\b", s):
        return "miami international airport mia miami fl"

    s = re.sub(r"\b(s\s*/\s*n|s\s*n|sem\s+numero)\b", "sn", s)
    s = re.sub(r"\bn[oº]?\s+(\d+)", r"\1", s)
    s = re.sub(r"(\b\d{1,5})\s*[a-z]\b", r"\1", s)
    s = re.sub(r"(\b\d{1,3})\s+(\d{3}\b)", r"\1\2", s)
    s = re.sub(r"\b(?:salas?|lojas?|cjs?|conjuntos?|blocos?|bl)\s+[a-z0-9\-/]+", " ", s)
    s = re.sub(r"\b\d+[oºaª]?\s*(?:andar|andares|floor|floors)\b", " ", s)

    s = re.sub(r"\b(bairro)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = " ".join(s.split())
    s = re.sub(r"\b(belo horizonte|goiania|recife|brasilia|go|mg|pe|df|sp)\s+\1\b", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()

def agrupar_numeros_proximos(lista_enderecos, tolerancia_numero=50):
    parsed = []
    for addr in lista_enderecos:
        m = re.search(r"^(.*?)\b(\d{1,5})\b(.*)$", addr)
        if m:
            prefix, num, suffix = m.group(1).strip(), int(m.group(2)), m.group(3).strip()
            parsed.append({"raw": addr, "prefix": prefix, "num": num, "suffix": suffix, "key": f"{prefix} | {suffix}"})
        else:
            parsed.append({"raw": addr, "prefix": None, "num": None, "suffix": None, "key": addr})

    groups = defaultdict(list)
    for p in parsed:
        groups[p["key"]].append(p)

    mapped = {}
    for _, items in groups.items():
        if len(items) == 1 or items[0]["num"] is None:
            for item in items:
                mapped[item["raw"]] = item["raw"]
        else:
            items_sorted = sorted(items, key=lambda x: x["num"])
            clusters = []
            for item in items_sorted:
                placed = False
                for cluster in clusters:
                    if abs(item["num"] - cluster[0]["num"]) <= tolerancia_numero:
                        cluster.append(item)
                        placed = True
                        break
                if not placed:
                    clusters.append([item])
            for cluster in clusters:
                rep_num = cluster[0]["num"]
                for item in cluster:
                    mapped[item["raw"]] = f"{item['prefix']} {rep_num} {item['suffix']}".strip()

    return [mapped[a] for a in lista_enderecos]

df["descricao_limpa"] = df["descricao"].apply(normalizar_texto)
df["origem_limpa"] = df["origem"].apply(pipeline_limpeza_completa)
df["destino_limpa"] = df["destino"].apply(pipeline_limpeza_completa)

enderecos_unicos = pd.concat([df["origem_limpa"], df["destino_limpa"]]).dropna().unique().tolist()
enderecos_agrupados = agrupar_numeros_proximos(enderecos_unicos, tolerancia_numero=50)

mapa_agrupamento = dict(zip(enderecos_unicos, enderecos_agrupados))
df["origem_consolidada"] = df["origem_limpa"].map(mapa_agrupamento)
df["destino_consolidado"] = df["destino_limpa"].map(mapa_agrupamento)
registrar_etapa("5. Limpeza de Endereços", t_step)

# ==========================================
# 6. EXPORTAÇÃO DOS LOCAIS ÚNICOS PARA CLASSIFICAÇÃO
# ==========================================
t_step = time.perf_counter()
print("[6/7] Exportando lista de locais consolidados...", flush=True)
locais_finais = sorted(pd.Series(enderecos_agrupados).unique())
df_locais = pd.DataFrame({"endereco_consolidado": locais_finais})

caminho_exportacao = "dados/locais_para_classificar.csv"
df_locais.to_csv(caminho_exportacao, index=False, encoding="utf-8")
print(f"   Gerado '{caminho_exportacao}' com {len(df_locais)} locais únicos.", flush=True)
registrar_etapa("6. Exportação de Locais", t_step)

# ==========================================
# 7. INTEGRAÇÃO COM LOCAIS CLASSIFICADOS E SCORE DE SUSPEITA
# ==========================================
t_step = time.perf_counter()
print("[7/7] Cruzando geolocalização, regras de política e calculando score...", flush=True)

caminho_classificados = "dados/locais_classificados.csv"
if not os.path.exists(caminho_classificados) and os.path.exists("locais_classificados.csv"):
    caminho_classificados = "locais_classificados.csv"

if os.path.exists(caminho_classificados):
    locais_classificados = pd.read_csv(caminho_classificados)

    df = df.merge(
        locais_classificados,
        how="left",
        left_on="origem_consolidada",
        right_on="endereco_consolidado"
    ).rename(columns={"categoria_local": "categoria_origem"})
    df.drop(columns=["endereco_consolidado"], errors="ignore", inplace=True)

    df = df.merge(
        locais_classificados,
        how="left",
        left_on="destino_consolidado",
        right_on="endereco_consolidado"
    ).rename(columns={"categoria_local": "categoria_destino"})
    df.drop(columns=["endereco_consolidado"], errors="ignore", inplace=True)

    df["categoria_origem"] = df["categoria_origem"].fillna("Outros")
    df["categoria_destino"] = df["categoria_destino"].fillna("Outros")

    # ------------------------------------------
    # VALIDAÇÕES CONTEXTUAIS DA POLÍTICA INTERNA
    # ------------------------------------------
    # Contexto 1: "aclara" na descrição + partida ou destino como "Escritório"
    menciona_aclara = df["descricao_limpa"].str.contains(r"\baclara\b", na=False)
    envolve_escritorio = (df["categoria_origem"] == "Escritório") | (df["categoria_destino"] == "Escritório")
    df["contexto_aclara_escritorio"] = menciona_aclara & envolve_escritorio

    df.loc[df["contexto_aclara_escritorio"], "categoria_justificativa_nlp"] = "Negócios"
    df.loc[df["contexto_aclara_escritorio"], "score_confianca_nlp"] = 1.0

    # Contexto 2 (Política de Jantar em Restaurante):
    padrao_jantar_refeicao = (
        r"\b(?:jantar|jantares|janta|dinner|cena|almoco|almocos|almuerzo|lunch|luch|"
        r"restaurante|restaurantes|restaurant|refeicao|comida)\b"
    )
    menciona_jantar_refeicao = df["descricao_limpa"].str.contains(padrao_jantar_refeicao, na=False, regex=True)
    envolve_restaurante = (df["categoria_origem"] == "Restaurante") | (df["categoria_destino"] == "Restaurante")
    df["contexto_jantar_restaurante"] = menciona_jantar_refeicao & envolve_restaurante

    df.loc[df["contexto_jantar_restaurante"], "categoria_justificativa_nlp"] = "Negócios"
    df.loc[df["contexto_jantar_restaurante"], "score_confianca_nlp"] = 1.0

    # ------------------------------------------
    # CÁLCULO DO SCORE DE SUSPEITA
    # ------------------------------------------
    df["score_suspeita"] = df["qtd_alertas_temporais"].astype(float)

    # Regra 1: Justificativa genérica ou ausente fora do horário comercial (+1.5 pontos)
    viagem_fora_horario = df["fim_de_semana"] | df["madrugada"] | df["feriado"]
    df.loc[
        viagem_fora_horario & df["categoria_justificativa_nlp"].isin(["Generica", "Sem Justificativa"]),
        "score_suspeita"
    ] += 1.5

    # Regra 2: Inconsistência semântica (cita viagem/hotel/voo, mas não passa por Hotel/Aeroporto) (+2.0 pontos)
    foi_para_hospedagem_ou_aero = (
        df["categoria_origem"].isin(["Hotel", "Aeroporto"])
        | df["categoria_destino"].isin(["Hotel", "Aeroporto"])
    )
    df.loc[
        (df["categoria_justificativa_nlp"] == "Viagem_Logistica") & ~foi_para_hospedagem_ou_aero,
        "score_suspeita"
    ] += 2.0

    # Regra 3: Viagem fora do horário comercial que não envolve locais autorizados (+2.0 pontos)
    locais_autorizados = ["Escritório", "Aeroporto", "Hotel", "Restaurante"]
    eh_local_autorizado = (
        df["categoria_origem"].isin(locais_autorizados)
        | df["categoria_destino"].isin(locais_autorizados)
    )
    df.loc[viagem_fora_horario & ~eh_local_autorizado, "score_suspeita"] += 2.0

    # Isenção final pela política de jantar em restaurante
    df.loc[df["contexto_jantar_restaurante"], "score_suspeita"] = 0.0

    # Exporta a Fila de Prioridade
    fila_prioridade = df.sort_values(by="score_suspeita", ascending=False)
    caminho_csv_fila = "dados/fila_prioridade_auditoria.csv"
    fila_prioridade.to_csv(caminho_csv_fila, index=False, encoding="utf-8")

    try:
        fila_prioridade.to_excel("dados/fila_prioridade_auditoria.xlsx", index=False)
        print("   Arquivos 'dados/fila_prioridade_auditoria.csv' e '.xlsx' salvos.", flush=True)
    except Exception:
        print(f"   Arquivo '{caminho_csv_fila}' salvo.", flush=True)

    registrar_etapa("7. Cruzamento e Exportação Final", t_step)

    # ==========================================
    # RELATÓRIO DE AVALIAÇÃO E DESEMPENHO
    # ==========================================
    tempo_total = time.perf_counter() - tempo_inicio_total
    print("==================================================", flush=True)
    print(f"RESUMO DE EXECUÇÃO (TEMPO TOTAL: {tempo_total:.2f}s)", flush=True)
    print("==================================================", flush=True)
    for etapa, dur in tempos_etapas.items():
        print(f"  - {etapa:<33}: {dur:.2f}s", flush=True)

    print("\n--- Distribuição das Categorias NLP (Pós-Regras) ---", flush=True)
    dist_nlp = df["categoria_justificativa_nlp"].value_counts()
    for cat, qtd in dist_nlp.items():
        pct = (qtd / len(df)) * 100
        print(f"  - {cat:<22}: {qtd:>4} corridas ({pct:.1f}%)", flush=True)

    print(f"  * Confiança média do modelo NLP : {df['score_confianca_nlp'].mean():.3f}", flush=True)
    print(f"  * Isenções Aclara + Escritório  : {df['contexto_aclara_escritorio'].sum()} corridas", flush=True)
    print(f"  * Isenções Jantar + Restaurante : {df['contexto_jantar_restaurante'].sum()} corridas", flush=True)

    print("\n--- Distribuição da Fila de Auditoria (Score) ---", flush=True)
    alta = (df["score_suspeita"] >= 3.0).sum()
    media = ((df["score_suspeita"] >= 1.5) & (df["score_suspeita"] < 3.0)).sum()
    baixa = (df["score_suspeita"] < 1.5).sum()
    print(f"  - Prioridade Alta  (Score >= 3.0): {alta:>4} corridas", flush=True)
    print(f"  - Prioridade Média (1.5 a 2.9)   : {media:>4} corridas", flush=True)
    print(f"  - Normal / Baixa   (Score < 1.5) : {baixa:>4} corridas", flush=True)
    print("==================================================", flush=True)

else:
    registrar_etapa("7. Verificação de Locais", t_step)
    print(f"Aguardando o preenchimento do arquivo '{caminho_classificados}' para calcular a fila de prioridade final.", flush=True)