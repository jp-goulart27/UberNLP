import pandas as pd
import re
import os
import json
import math
import time
import hashlib
import unicodedata
import requests

# ==========================================
# 1. CONFIGURAÇÃO DA API E DOS ARQUIVOS
# ==========================================
API_KEY = "SUA_CHAVE_AQUI"

CAMINHO_ENTRADA = "dados/locais_para_classificar.csv"
if not os.path.exists(CAMINHO_ENTRADA):
    CAMINHO_ENTRADA = "locais_para_classificar.csv"

CAMINHO_ESCRITORIOS = "dados/escritorios.csv"
if not os.path.exists(CAMINHO_ESCRITORIOS):
    CAMINHO_ESCRITORIOS = "escritorios.csv"

CAMINHO_CACHE_ESCRITORIOS = "dados/escritorios_coords.json"
CAMINHO_SAIDA = "dados/locais_classificados.csv"

# Raios de busca diferenciados por categoria
RAIO_ESCRITORIO_METROS = 100.0   # Raio ao redor dos escritórios oficiais da empresa
RAIO_HOTEL_AERO_METROS = 100.0    # Raio para hotéis e aeroportos
RAIO_RESTAURANTE_METROS = 50.0   # Raio menor para restaurantes (evita falsos positivos na mesma quadra)

# ==========================================
# 2. PADRÕES REGEX RÁPIDOS (PRÉ-FILTRO)
# ==========================================
PATRAO_AEROPORTO = (
    r"\b("
    r"aeroporto|airport|terminal|aerodrome|aeropuerto|base aerea|"
    r"cnf|gyn|bsb|rec|gru|cgh|vcp|sdu|gig|nat|mia|dca|scl|phl|jfk|yyz|slc|"
    r"confins|viracopos|congonhas|cumbica|galeao|guararapes|santa genoveva"
    r")\b"
)

PATRAO_HOTEL = (
    r"\b("
    r"hotel|hoteis|pousada|pousadas|resort|resorts|flat|flats|inn|suite|suites|hostel|hostels|"
    r"apparthotel|hospedagem|chale|chales|motel|"
    r"ibis|novotel|mercure|accor|hilton|marriott|bourbon|slaviero|blue tree|transamerica|"
    r"windsor|radisson|comfort|quality|hyatt|sheraton|intercontinental|fasano|tryp|tulip|"
    r"golden tulip|staybridge|caesar|gran melia|pullman|royal tulip|holiday inn|"
    r"ibis budget|ibis styles|adagio|nobile|bristol|intercity|verano|e-suite|e-suites"
    r")\b"
)

PATRAO_RESTAURANTE = (
    r"\b("
    r"restaurante|restaurantes|restaurant|churrascaria|pizzaria|bistro|trattoria|"
    r"cantina|osteria|steakhouse|temakeria|gastronomia|outback|coco bambu|madero|fogo de chao"
    r")\b"
)

# ==========================================
# 3. FUNÇÕES AUXILIARES DE TEXTO E DISTÂNCIA
# ==========================================
def normalizar_texto(texto):
    if pd.isna(texto):
        return ""
    s = str(texto).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r",\s*(brasil|brazil)\b.*$", "", s)
    s = re.sub(r"\b\d{5}\s?-?\s?\d{3}\b", " ", s)
    s = re.sub(r"\b\d{8}\b", " ", s)
    s = re.sub(r"\br\.\s*", "rua ", s)
    s = re.sub(r"\br\b\s+", "rua ", s)
    s = re.sub(r"\bav\.\s*", "avenida ", s)
    s = re.sub(r"\bav\b\s+", "avenida ", s)
    s = re.sub(r"\bn[oº\.]?\s*(\d+)", r"\1", s)
    s = re.sub(r"\bt\s*0+([1-9])", r"t \1", s)
    s = re.sub(r"\bt-([1-9])", r"t \1", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def calcular_distancia_metros(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * R * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


# ==========================================
# 4. BASE DE ESCRITÓRIOS DA EMPRESA
# ==========================================
def extrair_regras_base_escritorios(caminho_csv):
    if not os.path.exists(caminho_csv):
        return []

    df_esc = pd.read_csv(caminho_csv, sep=None, engine="python", encoding="utf-8-sig")
    regras = []

    for _, row in df_esc.iterrows():
        end_orig = str(row.get("endereco", "")).strip()
        cidade_orig = normalizar_texto(row.get("cidade", ""))
        if not end_orig or end_orig.lower() == "nan":
            continue

        parte_principal = re.split(r"\s-\s|,", end_orig)[0]
        parte_norm = normalizar_texto(parte_principal)
        vias = re.split(r"\s+(?=(?:rua|avenida|alameda|rodovia|travessa|praca)\b)", parte_norm)

        m_num = re.search(r",\s*(?:n[oº\.]?\s*)?(\d+)\b", end_orig, flags=re.IGNORECASE)
        if not m_num:
            m_num = re.search(r"\b(\d{2,5})\b", parte_principal)
        numero_ref = int(m_num.group(1)) if m_num else None

        for via in vias:
            via_limpa = re.sub(rf"\b{numero_ref}\b\s*$", "", via).strip() if numero_ref else via.strip()
            via_limpa = re.sub(r"\s+", " ", via_limpa).strip()
            if len(via_limpa) >= 5:
                regras.append({
                    "via_completa": via_limpa,
                    "numero": numero_ref,
                    "cidade": cidade_orig,
                })
    return regras


def geocodificar_endereco(endereco, api_key):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": endereco, "key": api_key}
    for _ in range(2):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                dados = r.json()
                if dados.get("status") == "OK" and dados.get("results"):
                    loc = dados["results"][0]["geometry"]["location"]
                    return float(loc["lat"]), float(loc["lng"])
                return None
        except requests.exceptions.RequestException:
            time.sleep(1.5)
    return None


def gerar_queries_geocode_escritorio(endereco):
    partes = re.split(r"(?=\b(?:Av\.?|Avenida|R\.?|Rua)\b)", endereco, flags=re.IGNORECASE)
    partes_vias = [p.strip() for p in partes if p.strip()]
    if len(partes_vias) >= 2 and "," in partes_vias[1]:
        resto = partes_vias[1].split(",", 1)[1]
        via_1 = partes_vias[0].rstrip(", ")
        via_2 = partes_vias[1].split(",", 1)[0].strip()
        return [f"{via_1}, {resto}", f"{via_2}, {resto}"]
    return [endereco]


def carregar_coordenadas_escritorios(caminho_csv, caminho_cache, api_key):
    if not os.path.exists(caminho_csv):
        return []

    with open(caminho_csv, "rb") as f:
        hash_atual = hashlib.md5(f.read()).hexdigest()

    if os.path.exists(caminho_cache):
        try:
            with open(caminho_cache, "r", encoding="utf-8") as f:
                cache = json.load(f)
            if cache.get("hash_csv") == hash_atual and cache.get("escritorios"):
                return cache["escritorios"]
        except Exception:
            pass

    df_esc = pd.read_csv(caminho_csv, sep=None, engine="python", encoding="utf-8-sig")
    pontos_escritorios = []

    for _, row in df_esc.iterrows():
        nome = str(row.get("nome_local", "Escritório")).strip()
        endereco = str(row.get("endereco", "")).strip()
        if not endereco or endereco.lower() == "nan":
            continue
        for query in gerar_queries_geocode_escritorio(endereco):
            coords = geocodificar_endereco(query, api_key)
            if coords:
                pontos_escritorios.append({
                    "nome_local": nome,
                    "endereco": query,
                    "lat": coords[0],
                    "lng": coords[1],
                })

    os.makedirs(os.path.dirname(caminho_cache) or ".", exist_ok=True)
    with open(caminho_cache, "w", encoding="utf-8") as f:
        json.dump({"hash_csv": hash_atual, "escritorios": pontos_escritorios}, f, ensure_ascii=False, indent=2)

    return pontos_escritorios


def match_escritorio_textual(endereco, regras_escritorios):
    end_norm = normalizar_texto(endereco)
    for regra in regras_escritorios:
        if regra["cidade"] and regra["cidade"] not in end_norm:
            continue
        if regra["via_completa"] in end_norm:
            if regra["numero"] is None:
                return True
            nums = [int(n) for n in re.findall(r"\b\d{1,5}\b", end_norm)]
            if not nums or any(abs(n - regra["numero"]) <= 60 for n in nums):
                return True
    return False


# ==========================================
# 5. CONSULTA GOOGLE PLACES API (NEW)
# ==========================================
def buscar_places_nearby(lat, lng, raio_metros, included_types, api_key):
    url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.displayName,places.types,places.primaryType"
    }
    payload = {
        "includedTypes": included_types,
        "maxResultCount": 5,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": float(raio_metros)
            }
        }
    }
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=8)
        if r.status_code == 200:
            return r.json().get("places", [])
    except Exception:
        pass
    return []


def classificar_endereco_completo(endereco, api_key, regras_escritorios, pontos_escritorios):
    if pd.isna(endereco) or not str(endereco).strip():
        return "Outros"

    texto = str(endereco).strip()

    # 1. Aeroporto (Regex)
    if re.search(PATRAO_AEROPORTO, texto, re.IGNORECASE):
        return "Aeroporto"

    # 2. Escritório da Empresa (Match textual contra escritorios.csv)
    if match_escritorio_textual(texto, regras_escritorios):
        return "Escritório"

    # 3. Hotel (Regex)
    if re.search(PATRAO_HOTEL, texto, re.IGNORECASE):
        return "Hotel"

    # 4. Restaurante explícito no endereço (Regex)
    if re.search(PATRAO_RESTAURANTE, texto, re.IGNORECASE):
        return "Restaurante"

    # 5. Geocodificação para checagem por raio na API
    if not api_key or api_key == "SUA_CHAVE_API_AQUI":
        return "Outros"

    coords = geocodificar_endereco(texto, api_key)
    if not coords:
        return "Outros"

    lat, lng = coords

    # 5a. Escritório da Empresa por raio geodésico (RAIO_ESCRITORIO_METROS)
    for esc in pontos_escritorios:
        dist = calcular_distancia_metros(lat, lng, esc["lat"], esc["lng"])
        if dist <= RAIO_ESCRITORIO_METROS:
            return "Escritório"

    # 5b. Hotel / Aeroporto via Places Nearby (RAIO_HOTEL_AERO_METROS = 50m)
    tipos_hotel_aero = [
        "airport", "international_airport",
        "lodging", "hotel", "resort_hotel", "extended_stay_hotel",
        "guest_house", "bed_and_breakfast", "hostel", "inn", "motel"
    ]
    places_ha = buscar_places_nearby(lat, lng, RAIO_HOTEL_AERO_METROS, tipos_hotel_aero, api_key)
    for p in places_ha:
        p_types = {t.lower() for t in p.get("types", [])}
        if p.get("primaryType"):
            p_types.add(p["primaryType"].lower())
        if p_types.intersection({"airport", "international_airport"}):
            return "Aeroporto"
        if p_types.intersection(set(tipos_hotel_aero)):
            return "Hotel"

    # 5c. Restaurante via Places Nearby com RAIO MENOR (RAIO_RESTAURANTE_METROS = 20m)
    tipos_restaurante = [
        "restaurant", "steak_house", "brazilian_restaurant", "italian_restaurant",
        "japanese_restaurant", "seafood_restaurant", "pizza_restaurant",
        "hamburger_restaurant", "barbecue_restaurant", "brunch_restaurant",
        "cafe", "bar", "pub"
    ]
    places_rest = buscar_places_nearby(lat, lng, RAIO_RESTAURANTE_METROS, tipos_restaurante, api_key)
    for p in places_rest:
        p_types = {t.lower() for t in p.get("types", [])}
        if p.get("primaryType"):
            p_types.add(p["primaryType"].lower())
        if p_types.intersection(set(tipos_restaurante)):
            return "Restaurante"

    return "Outros"


# ==========================================
# 6. EXECUÇÃO E EXPORTAÇÃO
# ==========================================
if __name__ == "__main__":
    df_locais = pd.read_csv(CAMINHO_ENTRADA)
    coluna_endereco = df_locais.columns[0]

    regras_esc = extrair_regras_base_escritorios(CAMINHO_ESCRITORIOS)
    pontos_esc = (
        carregar_coordenadas_escritorios(CAMINHO_ESCRITORIOS, CAMINHO_CACHE_ESCRITORIOS, API_KEY)
        if API_KEY and API_KEY != "SUA_CHAVE_API_AQUI"
        else []
    )

    total = len(df_locais)
    categorias = []
    print(f"Classificando {total} locais (Raio Hotel/Aero: {RAIO_HOTEL_AERO_METROS}m | Raio Restaurante: {RAIO_RESTAURANTE_METROS}m)...")

    for idx, endereco in enumerate(df_locais[coluna_endereco], start=1):
        cat = classificar_endereco_completo(endereco, API_KEY, regras_esc, pontos_esc)
        categorias.append(cat)
        if idx % 50 == 0 or idx == total:
            print(f"Processados {idx}/{total} locais...")

    df_locais.rename(columns={coluna_endereco: "endereco_consolidado"}, inplace=True)
    df_locais["categoria_local"] = categorias

    os.makedirs("dados", exist_ok=True)
    df_locais.to_csv(CAMINHO_SAIDA, index=False, encoding="utf-8")
    df_locais.to_csv("locais_classificados.csv", index=False, encoding="utf-8")

    print("\n--- RESUMO DA CLASSIFICAÇÃO ---")
    print(df_locais["categoria_local"].value_counts().to_string())
    print(f"\nArquivo salvo em '{CAMINHO_SAIDA}'.")
