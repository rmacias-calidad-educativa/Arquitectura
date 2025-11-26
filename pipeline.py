# pipeline.py
import os
import json
import re
import time
import string
import copy
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- Modelos ----------
MODEL_PARSE    = "gpt-5-mini"
MODEL_CLASSIFY = "gpt-5-mini"
RUBRIC_MODEL   = "gpt-5-mini"

# ========= Cabeceras y pistas (agnóstico al área) =========
HEAD_TERMS = [r"Unidad", r"M[oó]dulo", r"Bloque"]  # sinónimos válidos
HEAD_ALT = "(?:" + "|".join(HEAD_TERMS) + ")"
ACCEPT_HINTS_BASE = [
    "desempeño precisado", "desempeno precisado",
    "evidencia de logro", "producto sugerido",
    "instrumento sugerido", "criterios de evaluación",
    "aprendizajes esperados", "resultado de aprendizaje",
    "estándares", "estandares", "competencia", "competencias"
]

# =========================================================
# Cliente OpenAI
# =========================================================
def get_client() -> OpenAI:
    """
    Devuelve un cliente OpenAI reutilizable.
    Usa OPENAI_API_KEY desde las variables de entorno.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No se encontró OPENAI_API_KEY en las variables de entorno. "
            "Configúrala antes de ejecutar el pipeline."
        )
    return OpenAI(api_key=api_key)

def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S)
    return s.strip()

def call_openai_json(client: OpenAI, model: str, system: str, user: str, retries: int = 1) -> dict:
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system + " Devuelve SOLO un objeto JSON válido."},
                    {"role": "user",   "content": user}
                ],
                timeout=60.0,
            )
            content = _strip_code_fences(resp.choices[0].message.content or "{}")
            try:
                return json.loads(content)
            except Exception:
                m = re.search(r"\{.*\}", content, flags=re.S)
                if m:
                    return json.loads(m.group(0))
                return {}
        except Exception as e:
            if attempt >= retries:
                print(f"⚠️ Error LLM: {e}")
                return {}
            time.sleep(1.2 * (attempt + 1))
    return {}

# =========================================================
# Utilidades texto
# =========================================================
def initials(texto: str) -> str:
    acc = []
    for word in (texto or "").strip().split():
        for ch in word:
            if ch.isalpha():
                acc.append(ch.upper())
                break
    return "".join(acc) or "X"

def _shorten_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    s = re.sub(r"\s+", " ", str(text)).strip()
    if len(s) <= max_chars:
        return s
    s_cut = s[: max_chars - 3]
    if " " in s_cut:
        s_cut = s_cut.rsplit(" ", 1)[0]
    return s_cut + "..."

def make_desempeno_title(dp_text: str, maybe_title: str = "", max_chars: int = 80) -> str:
    base = (maybe_title or "").strip()
    if base:
        return _shorten_text(base, max_chars)
    return _shorten_text(dp_text, max_chars)

def make_desempeno_short(dp_text: str, max_chars: int = 200) -> str:
    return _shorten_text(dp_text, max_chars)

# =========================================================
# Mapeo de grados y bandas
# =========================================================
GRADE_NAMES = {
    -2: "Prejardín", -1: "Jardín", 0: "Transición",
    1: "Primero", 2: "Segundo", 3: "Tercero", 4: "Cuarto",
    5: "Quinto", 6: "Sexto", 7: "Séptimo", 8: "Octavo",
    9: "Noveno", 10: "Décimo", 11: "Once",
}
BANDS_ORDER = ["Descubrimiento","Exploración","Construcción","Conexión"]

def grade_name(n: int) -> str:
    try:
        return GRADE_NAMES[int(n)]
    except Exception:
        return str(n)

def band_for_grade(n: int) -> str:
    n = int(n)
    if n <= 0: return "Descubrimiento"
    if 1 <= n <= 4: return "Exploración"
    if 5 <= n <= 8: return "Construcción"
    if 9 <= n <= 11: return "Conexión"
    return "Exploración"

def _band_index(txt: str) -> int:
    t = (txt or "").strip().lower()
    for i, b in enumerate(BANDS_ORDER):
        if b.lower() in t:
            return i
    return -1

def stage_allows_grade(stage: str, grado: int) -> bool:
    if not stage:
        return True
    s = stage.strip()
    s_low = s.lower()
    if "toda" in s_low:
        return True
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)", s)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return lo <= int(grado) <= hi
    if re.search(r"\b(a|y)\b", s_low):
        parts = re.split(r"\s+(?:a|y)\s+", s, flags=re.I)
        idxs = [_band_index(p) for p in parts if _band_index(p) >= 0]
        if not idxs:
            return True
        lo, hi = min(idxs), max(idxs)
        g_idx = _band_index(band_for_grade(grado))
        return lo <= g_idx <= hi
    return band_for_grade(grado).lower() in s_low

# =========================================================
# Catálogo de Componentes
# =========================================================
COMPONENTS_CATALOG: Dict[str, List[Dict[str, Any]]] = {
    "COMUNICACIÓN": [
        {"code": "E",  "name": "Producción Escrita",   "stage": "Todas"},
        {"code": "A",  "name": "Comprensión Auditiva", "stage": "Todas"},
        {"code": "L",  "name": "Comprensión Lectora",  "stage": "Todas"},
        {"code": "O",  "name": "Producción Oral",      "stage": "Todas"},
    ],
    "FILOSOFÍA": [
        {"code": "PAN","name": "Pensamiento Antropológico",  "stage": "Conexión 10 - 11"},
        {"code": "PEP","name": "pensamiento Epistemológico", "stage": "Conexión 10 - 11"},
        {"code": "PLO","name": "pensamiento Lógico",         "stage": "Conexión 10 - 11"},
        {"code": "PET","name": "pensamiento Ético",          "stage": "Conexión 10 - 11"},
        {"code": "PES","name": "pensamiento Estético",       "stage": "Conexión 10 - 11"},
        {"code": "PON","name": "pensamiento Ontológico",     "stage": "Conexión 10 - 11"},
    ],
    "INGLÉS": [
        {"code": "V",  "name": "Vocabulary Building",       "stage": "Descubrimiento"},
        {"code": "F",  "name": "Fluency",                   "stage": "Descubrimiento"},
        {"code": "RC", "name": "Reading and Comprehension", "stage": "Descubrimiento"},
        {"code": "S",  "name": "Speaking",                  "stage": "Exploración a Conexión"},
        {"code": "L",  "name": "Listening",                 "stage": "Exploración a Conexión"},
        {"code": "W",  "name": "Writing",                   "stage": "Exploración a Conexión"},
        {"code": "R",  "name": "Reading",                   "stage": "Exploración a Conexión"},
    ],
    "CIENCIAS SOCIALES": [
        {"code": "RI",  "name": "Relaciones Interpersonales", "stage": "Descubrimiento"},
        {"code": "CE",  "name": "Convivir con el Entorno",    "stage": "Descubrimiento"},
        {"code": "H",   "name": "History",                    "stage": "Exploración a Conexión"},
        {"code": "GEO", "name": "Geography",                  "stage": "Exploración a Conexión"},
        {"code": "E",   "name": "Economy",                    "stage": "Exploración a Conexión"},
        {"code": "C",   "name": "Citizenship",                "stage": "Exploración a Conexión"},
        {"code": "S",   "name": "Society",                    "stage": "Exploración a Conexión"},
        {"code": "P",   "name": "Politics",                   "stage": "Exploración a Conexión"},
    ],
    "DESARROLLO PERSONAL": [
        {"code": "IAC","name": "Identidad y autoconocimiento",                      "stage": "Descubrimiento"},
        {"code": "IE", "name": "Inteligencia emocional",                            "stage": "Descubrimiento"},
        {"code": "RI", "name": "Relaciones interpersonales",                        "stage": "Descubrimiento"},
        {"code": "CE", "name": "Convivir con el entorno",                           "stage": "Descubrimiento"},
        {"code": "IECP","name":"Inteligencia emocional y comunicación positiva",    "stage": "Exploración"},
        {"code": "ARTD","name":"Autonomía, responsabilidad y toma de decisiones",   "stage": "Exploración"},
        {"code": "PCP","name":"Plan de crecimiento personal",                       "stage": "Exploración"},
        {"code": "ESRC","name":"Enfoque en soluciones y relaciones constructivas.", "stage": "Exploración"},
        {"code": "ACAR","name":"Autoconciencia y autorregulación.",                 "stage": "Construcción"},
        {"code": "IAP","name":"Identidad y autonomía personal.",                    "stage": "Construcción"},
        {"code": "RCA","name":"Relaciones conscientes y afectividad.",              "stage": "Construcción"},
        {"code": "CAD","name":"Ciudadanía activa y digital.",                       "stage": "Construcción"},
        {"code": "AREA","name":"Autorregulación y estrategias de afrontamiento.",   "stage": "Conexión"},
        {"code": "IPA","name":"Identidad, propósito y autonomía personal.",         "stage": "Conexión"},
        {"code": "RERA","name":"Relaciones éticas y responsabilidad afectiva.",     "stage": "Conexión"},
        {"code": "CCDR","name":"Ciudadanía crítica, digital y relacional.",         "stage": "Conexión"},
    ],
    "CIENCIAS NATURALES": [
        {"code": "SPC","name":"Pensamiento Científico", "stage":"Todas"},
        {"code": "SPD","name":"Pensamiento de Diseño",  "stage":"Todas"},
        {"code": "SCA","name":"Ciudadanía Ambiental",   "stage":"Todas"},
    ],
    "TECNOLOGÍA": [
        {"code": "TPC","name":"Pensamiento Computacional", "stage":"Exploración a Conexión"},
        {"code": "TPD","name":"Pensamiento de Diseño",     "stage":"Exploración a Conexión"},
        {"code": "TRP","name":"Robotica y programación",   "stage":"Exploración a Conexión"},
    ],
    "ED. ARTÍSTICA": [
        {"code": "CDE", "name":"Cultura y desarrollo estético",                "stage":"Todas"},
        {"code": "EIC", "name":"Experimentación, imaginación y creación",      "stage":"Descubrimiento"},
        {"code": "CECS","name":"Creatividad, expresión y construcción simbólica . ", "stage":"Exploración a Conexión"},
    ],
    "MÚSICA": [
        {"code": "MDE", "name":"Cultura y desarrollo estético",                "stage":"Todas"},
        {"code": "MIC", "name":"Experimentación, imaginación y creación",      "stage":"Descubrimiento"},
        {"code": "MECS","name":"Creatividad, expresión y construcción simbólica . ", "stage":"Exploración a Conexión"},
    ],
    "ED. FÍSICA": [
        {"code": "CC",  "name":"Conciencia y Control Corporal.",                        "stage":"Descubrimiento"},
        {"code": "CVHM","name":"Coordinación visomotora y habilidades manipulativas.", "stage":"Descubrimiento"},
        {"code": "SC",  "name":"Salud y cuidado personal.",                            "stage":"Descubrimiento"},
        {"code": "CC",  "name":"Conciencia y Control Corporal.",                        "stage":"Exploración"},
        {"code": "CVHM","name":"Coordinación visomotora y habilidades manipulativas.", "stage":"Exploración"},
        {"code": "SC",  "name":"Salud y cuidado personal.",                            "stage":"Exploración"},
        {"code": "DST", "name":"Deporte, salud y tecnología.",                         "stage":"Construcción y Conexión"},
        {"code": "DAFC","name":"Deporte, actividad física y ciencia.",                 "stage":"Construcción y Conexión"},
        {"code": "DCA", "name":"Deporte y ciudadanía activa.",                         "stage":"Construcción y Conexión"},
        {"code": "RDPE","name":"Rendimiento deportivo vs Prescripción del ejercicio.", "stage":"Construcción y Conexión"},
        {"code": "FBE", "name":"Fisiología básica del ejercicio.",                     "stage":"Construcción y Conexión"},
        {"code": "FA",  "name":"Fisiología aplicada.",                                 "stage":"Construcción y Conexión"},
        {"code": "DCC", "name":"Deporte, ciencia y cuerpo.",                           "stage":"Construcción y Conexión"},
        {"code": "CFRS","name":"Cultura física y redes sociales.",                     "stage":"Construcción y Conexión"},
        {"code": "DAFAD","name":"Deporte, actividad física y analisis de datos",       "stage":"Construcción y Conexión"},
        {"code": "PI",  "name":"Proyecto de impacto",                                  "stage":"Conexión"},
        {"code": "AFPV","name":"Actividad física y proyecto de vida",                  "stage":"Conexión"},
    ],
    "MATEMÁTICA": [
        {"code": "INC","name":"Ideas de número y cambio ", "stage":"Descubrimiento"},
        {"code": "EME","name":"Entiendo mi entorno",        "stage":"Descubrimiento"},
        {"code": "PN", "name":"Pensamiento numérico",       "stage":"Todas"},
        {"code": "PV", "name":"Pensamiento variacional",    "stage":"Todas"},
        {"code": "PE", "name":"Pensamiento espacial",       "stage":"Todas"},
        {"code": "PA", "name":"Pensamiento aleatorio",      "stage":"Todas"},
    ],
    "INDAGACIÓN": [
        {"code": "IN", "name":"Primeros pasos en indagación",                        "stage":"Descubrimiento"},
        {"code": "EX", "name":"Exploración inicial",                                  "stage":"Descubrimiento"},
        {"code": "CM", "name":"Comunicación de hallazgos",                            "stage":"Descubrimiento"},
        {"code": "IN", "name":"Indagación guiada",                                    "stage":"Exploración"},
        {"code": "EX", "name":"Exploración estructurada",                             "stage":"Exploración"},
        {"code": "CM", "name":"Comunicación de hallazgos",                            "stage":"Exploración"},
        {"code": "IN", "name":"Construcción argumentada de preguntas investigativas", "stage":"Construcción"},
        {"code": "EX", "name":"Diseño y ejecución de estrategias de exploración",     "stage":"Construcción"},
        {"code": "CM", "name":"Socialización de hallazgos",                           "stage":"Construcción"},
        {"code": "PR", "name":"Planteamiento crítico de problemas",                   "stage":"Conexión"},
        {"code": "MT", "name":"Diseño y análisis metodológico",                       "stage":"Conexión"},
        {"code": "DV", "name":"Divulgación académica y social de resultados",         "stage":"Conexión"},
    ],
}

def allowed_components_for(area: str, grado_num: int) -> List[Dict[str, Any]]:
    comps = COMPONENTS_CATALOG.get(area, [])
    out = []
    for c in comps:
        if stage_allows_grade(c.get("stage",""), grado_num):
            c2 = dict(c)
            c2["stage_label"] = grade_name(grado_num)
            out.append(c2)
    return out

# =========================================================
# Prompts de parseo
# =========================================================
PARSE_SYSTEM = (
    "Eres experto en evaluación curricular y normalizador de evaluaciones. "
    "Tu salida DEBE ser SOLO JSON válido. "
    "Extrae la UNIDAD (si está) y una lista de DESEMPEÑOS. "
    "Para cada desempeño, genera instancias verificadoras OBSERVACIONALES, sin cuantificar, "
    "manteniendo el mismo idioma del texto fuente (si el texto está en inglés, escribe en inglés; "
    "si está en español, escribe en español)."
)

PARSE_RULES_TMPL = """
Reglas para Instancias Verificadoras (observacionales):
- Observacionales, redactadas en positivo y concretas.
- Sin cuantificar (evita números, 'al menos', 'porcentaje', 'n veces').
- Una acción principal por ítem (máx. 2 cláusulas).
- Verbo observable al inicio (p. ej., 'identifica', 'describe', 'relaciona', 'organiza', 'recita', 'lee', 'escribe').
- Sin microdetalles ni criterios de corrección dentro del ítem.
- Genera exactamente {N} instancias verificadoras por cada desempeño, diferenciadas entre sí (no repitas la misma acción con sinónimos).
"""

PARSE_USER_TMPL_PREFIX = (
    "json\nConstruye un objeto con la forma exacta:\n"
    "{\n  \"unidad\": <número o null>,\n  \"desempenos\": [\n    {\n"
    "      \"codigo\": \"<id detectado o genera D1..DN si falta>\",\n"
    "      \"titulo\": \"<si existe>\",\n"
    "      \"desempeno_precisado\": \"<texto en el mismo idioma del texto fuente; no traduzcas>\",\n"
    "      \"Instancia verificadora\": [\"<instancia verificadora observacional>\", ...]\n"
    "    }\n  ]\n}\n\n"
)

def make_parse_prompt(raw_text: str, n: int) -> str:
    rules = PARSE_RULES_TMPL.format(N=int(n))
    return PARSE_USER_TMPL_PREFIX + rules + "\nTEXTO FUENTE:\n" + (raw_text or "")

# =========================================================
# Detección IA de cabeceras de UNIDAD
# =========================================================
UNITS_SYSTEM = (
    "Eres analista de documentos curriculares. "
    "Identifica cabeceras top-level de UNIDADES que aparezcan como:\n"
    "- 'Unidad 1' o 'Unidad 1 y 2' (sin dos puntos), y/o\n"
    "- 'Unidad 1: Título o pregunta...' (con dos puntos).\n"
    "No incluyas subapartados internos como 'Exploración y análisis'. "
    "Devuelve SOLO JSON."
)

UNITS_USER_TMPL = (
    "json\nDevuelve:\n{\n"
    "  \"unidades\": [\n"
    "     {\"numero_inicial\": <int>, \"numero_final\": <int|null>, \"header_linea\": \"<texto exacto de la línea>\"}\n"
    "  ]\n}\n\nTEXTO FUENTE:\n"
)

def ai_detect_top_level_units(raw_text: str) -> List[Dict[str, Any]]:
    client = get_client()
    USER = UNITS_USER_TMPL + (raw_text or "")
    data = call_openai_json(client, MODEL_PARSE, UNITS_SYSTEM, USER, retries=1)
    top = data.get("unidades") or data.get("top_level") or []
    cleaned = []
    for x in top:
        try:
            ni = int(x.get("numero_inicial"))
        except Exception:
            continue
        nf = x.get("numero_final", None)
        if nf is not None:
            try:
                nf = int(nf)
            except Exception:
                nf = None
        header = (x.get("header_linea") or x.get("header") or "").strip() or None
        cleaned.append({"numero_inicial": ni, "numero_final": nf, "header": header})
    cleaned.sort(key=lambda d: d["numero_inicial"])
    return cleaned

def _find_header_index(txt: str, numero: int, header: str | None) -> int | None:
    if header:
        patt = re.escape(header).replace(r"\ ", r"\s+")
        m = re.search(patt, txt, flags=re.I)
        if m:
            return m.start()
    pat = re.compile(
        rf'(?ms)^[ \t]*["“”]?[ \t]*{HEAD_ALT}[ \t]+{numero}(?:[ \t]*y[ \t]*\d+)?[ \t]*(?::[^\n]*|)(?:\n|[ ]{{2,}})'
    )
    m = pat.search(txt)
    return m.start() if m else None

INTERNAL_SUBSECTION_HINTS = [
    "exploración y análisis",
    "producción y comunicación",
    "apreciación",
    "creación, expresión y socialización",
    "exploración cultural y visual",
    "exploración cultural y dramatúrgica",
]

def _has_any(haystack: str, needles: list) -> bool:
    hay = haystack.lower()
    return any(n in hay for n in needles)

def regex_units_no_colon_pairs(text: str):
    txt = text or ""
    pat = re.compile(
        rf'(?ms)^[ \t]*["“”]?[ \t]*{HEAD_ALT}[ \t]+(\d+)(?:[ \t]*y[ \t]*(\d+))?(?![ \t]*:)[ \t]*(?:\n|[ ]{{2,}})'
    )
    hits = [(m.start(), int(m.group(1)), int(m.group(2)) if m.group(2) else None) for m in pat.finditer(txt)]
    if not hits:
        return []
    segments = []
    for i, (pos, u1, u2) in enumerate(hits):
        end = hits[i+1][0] if i+1 < len(hits) else len(txt)
        chunk = txt[pos:end].strip()
        if u2 is not None:
            segments.append((u1, chunk))
            segments.append((u2, chunk))
        else:
            segments.append((u1, chunk))
    by_unit = {}
    for u, ch in segments:
        by_unit.setdefault(u, ch)
    return sorted(by_unit.items(), key=lambda kv: kv[0])

def regex_units_with_colon_smart(text: str):
    txt = text or ""
    header_re = re.compile(rf'(?m)^[ \t]*["“”]?[ \t]*{HEAD_ALT}[ \t]+(\d+)[ \t]*:[ \t]*(.*)$')
    results = []
    for m in header_re.finditer(txt):
        header_tail = (m.group(2) or "").strip()
        tail_l = header_tail.lower()
        if _has_any(tail_l, INTERNAL_SUBSECTION_HINTS):
            continue
        accept = ("?" in header_tail)
        if not accept:
            pos = m.end()
            window = txt[pos:pos+1200].lower()
            if _has_any(window, ACCEPT_HINTS_BASE):
                accept = True
        if accept:
            try:
                u = int(m.group(1))
            except Exception:
                continue
            results.append((u, m.start()))
    if not results:
        return []
    results.sort(key=lambda t: t[1])
    chunks = []
    for i, (u, start) in enumerate(results):
        end = results[i+1][1] if i+1 < len(results) else len(txt)
        chunk = txt[start:end].strip()
        chunks.append((u, chunk))
    return chunks

def split_unidades_ai_best(big_text: str) -> List[Tuple[Optional[int], str]]:
    txt = big_text or ""
    if not txt.strip():
        return [(None, txt)]
    units = ai_detect_top_level_units(txt)
    starts: List[Tuple[Dict[str, Any], int]] = []
    for u in units:
        idx = _find_header_index(txt, u["numero_inicial"], u.get("header"))
        if idx is not None:
            starts.append((u, idx))
    if not starts:
        pat = re.compile(
            r'(?ms)^[ \t]*"?[ \t]*Unidad[ \t]+(\d+)(?:[ \t]*y[ \t]*\d+)?(?![ \t]*:)[ \t]*(?:\n|[ ]{2,})'
        )
        hits = [({"numero_inicial": int(m.group(1)), "numero_final": None, "header": None}, m.start()) for m in pat.finditer(txt)]
        if not hits:
            return [(None, txt)]
        starts = hits
    starts.sort(key=lambda t: t[1])
    blocks: List[Tuple[Optional[int], str]] = []
    for i, (info, pos) in enumerate(starts):
        end = starts[i+1][1] if i+1 < len(starts) else len(txt)
        chunk = txt[pos:end].strip()
        lo = int(info["numero_inicial"])
        hi = int(info["numero_final"]) if info.get("numero_final") else lo
        for u in range(lo, hi+1):
            blocks.append((u, chunk))
    return blocks

def split_unidades_combined(big_text: str) -> List[Tuple[Optional[int], str]]:
    txt = big_text or ""
    a = split_unidades_ai_best(txt)
    if a and any(isinstance(u, int) for u, _ in a):
        return a
    b = regex_units_with_colon_smart(txt)
    if b:
        return b
    return [(1, txt)]

def split_unidades_dispatch(txt: str) -> List[Tuple[Optional[int], str]]:
    return split_unidades_combined(txt)

# =========================================================
# Parser y normalizador
# =========================================================
def parse_unidad_text(raw_text: str, n_instancias_exact: int) -> Dict[str, Any]:
    client = get_client()
    user_prompt = make_parse_prompt(raw_text, n_instancias_exact)
    data = call_openai_json(client, MODEL_PARSE, PARSE_SYSTEM, user_prompt, retries=1)
    if not data:
        data = {"unidad": None, "desempenos": []}
    out = []
    for i, d in enumerate(data.get("desempenos", []) or [], start=1):
        cod = (d.get("codigo") or f"D{i}").strip()
        titulo = (d.get("titulo") or "").strip()
        dp = (d.get("desempeno_precisado") or d.get("desempeño_precisado") or "").strip()
        ivs = d.get("Instancia verificadora") or d.get("instancias verificadoras") or d.get("evidencias") or []
        ivs = [str(x).strip() for x in ivs if str(x).strip()]
        out.append({
            "codigo": cod,
            "titulo": titulo,
            "desempeno_precisado": dp,
            "instancias": ivs
        })
    return {"unidad": data.get("unidad", None), "desempenos": out}

# =========================================================
# Ajuste de IV
# =========================================================
IVS_PER_DES_DEFAULT = 2
IVS_PER_DES_BY_CODE: Dict[str, int] = {
    # Opcional: overrides por código de desempeño
}

def _override_target_from_map(code: str, title: str = "") -> Optional[int]:
    code = (code or "").strip()
    title_l = (title or "").lower()
    if code in IVS_PER_DES_BY_CODE:
        try:
            return int(IVS_PER_DES_BY_CODE[code])
        except Exception:
            return None
    for k, v in IVS_PER_DES_BY_CODE.items():
        k_l = (k or "").lower().strip()
        if k_l and (k_l in code.lower() or k_l in title_l):
            try:
                return int(v)
            except Exception:
                return None
    return None

def ivs_per_des_by_unit_rule(n_desempenos: int) -> int:
    """
    Regla original:
    1–2 ⇒ 4; 3 ⇒ 3; 4+ ⇒ 2 (IV por desempeño).
    Se usa SOLO si el usuario no define explícitamente la cantidad.
    """
    if n_desempenos <= 2:
        return 4
    if n_desempenos == 3:
        return 3
    return 2

IV_GEN_SYSTEM = (
    "Eres experto en evaluación curricular y generador de INSTANCIAS VERIFICADORAS observacionales. "
    "Responde SOLO JSON válido. Mantén el mismo idioma del desempeño proporcionado: "
    "si está en inglés, escribe las instancias en inglés; si está en español, escribe en español. "
    "Redacción positiva, concreta, sin cuantificar, con una acción principal por ítem."
)

def llm_generate_ivs_for_desempeno(desempeno_text: str, n: int) -> List[str]:
    if n <= 0:
        return []
    client = get_client()
    user = "json\n" + json.dumps({
        "instrucciones": [
            f"Genera EXACTAMENTE {int(n)} instancias verificadoras (lista JSON de strings).",
            "Observacionales, positivas, sin números/porcentajes/veces.",
            "Una acción principal por ítem (máx. 2 cláusulas).",
        ],
        "desempeno_precisado": (desempeno_text or "").strip(),
        "formato_esperado": {"instancias": ["..."]}
    }, ensure_ascii=False)
    data = call_openai_json(client, MODEL_PARSE, IV_GEN_SYSTEM, user, retries=1)
    ivs = data.get("instancias") or []
    ivs = [str(x).strip() for x in ivs if str(x).strip()]
    while len(ivs) < n and ivs:
        ivs.append(ivs[len(ivs) % max(1, len(ivs))])
    return ivs[:n]

def enforce_iv_targets_by_unit(
    desempenos: List[Dict[str,Any]],
    ivs_per_des_user: Optional[int] = None
) -> List[Dict[str,Any]]:
    """
    Ajusta la cantidad de instancias por desempeño.

    - Si ivs_per_des_user no es None, se fuerza esa cantidad para TODOS los desempeños
      (salvo que quieras respetar overrides manuales, que aquí se ignoran).
    - Si es None, se usa la regla por unidad (ivs_per_des_by_unit_rule) + overrides.
    """
    n_des = len(desempenos)
    if n_des == 0:
        return desempenos

    base_target = ivs_per_des_by_unit_rule(n_des)

    for d in desempenos:
        code = d.get("codigo","")
        title = d.get("titulo","")

        if ivs_per_des_user is not None:
            target = int(ivs_per_des_user)
        else:
            override = _override_target_from_map(code, title)
            if override is not None:
                target = int(override)
            else:
                target = int(base_target)

        have = len(d.get("instancias", []))
        if have < target:
            extra = llm_generate_ivs_for_desempeno(
                d.get("desempeno_precisado",""),
                target - have
            )
            d.setdefault("instancias", []).extend(extra)
        elif have > target:
            d["instancias"] = d["instancias"][:target]

    return desempenos

# =========================================================
# Clasificador de componentes
# =========================================================
def _heuristic_components(area: str, texto: str, allowed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not allowed:
        return []
    t = (texto or "").lower()
    codes: List[str] = []

    area_up = (area or "").strip().upper()
    if area_up in ("COMUNICACIÓN", "COMUNICACION"):
        # Producción Escrita
        if any(w in t for w in ["escribe", "escrit", "trazo", "garabato", "graf", "dibujo", "pseudoletra"]):
            codes.append("E")
        # Comprensión Auditiva
        if any(w in t for w in ["escucha", "escuchar", "instruccion", "instrucción", "oye", "oír", "auditiv", "canción", "ronda"]):
            codes.append("A")
        # Comprensión Lectora
        if any(w in t for w in ["lee", "lectura", "cuento", "libro", "texto visual", "imagen"]):
            codes.append("L")
        # Producción Oral
        if any(w in t for w in ["habla", "hablar", "oral", "narra", "narrar", "expresa", "convers", "contar"]):
            codes.append("O")

    comps: List[Dict[str, Any]] = []
    for code in codes:
        c = next((c for c in allowed if c.get("code", "").upper() == code.upper()), None)
        if c and c not in comps:
            comps.append(c)
        if len(comps) >= 3:
            break
    return comps

def llm_pick_components_for_instance(
    area: str,
    grado_num: int,
    ev_text: str,
    desempeno: str,
    max_k: int = 3
) -> List[Dict[str,str]]:
    allowed = allowed_components_for(area, grado_num)
    if not allowed:
        return []
    allowed_names = [c["name"] for c in allowed]
    allowed_codes = [c["code"] for c in allowed]
    SYSTEM = (
        "Eres un clasificador de componentes curriculares. "
        "Devuelve SOLO JSON válido con componentes de la lista permitida. "
        "Si no hay relación clara, inicialmente puedes dejar la lista vacía, "
        "pero el sistema aplicará una heurística para evitar filas sin componente."
    )
    USER = "json\n" + json.dumps({
        "instrucciones": [
            "Lee la instancia verificadora y/o el desempeño.",
            f"Elige de 1 a {max_k} componentes EXPLÍCITOS de la lista permitida.",
            "Puedes devolver por NOMBRE o por CÓDIGO, pero debe existir en la lista."
        ],
        "permitidos": {"nombres": allowed_names, "codigos": allowed_codes},
        "entrada": {"instancia": ev_text, "desempeno": desempeno},
        "formato_esperado": {"componentes": []}
    }, ensure_ascii=False)
    client = get_client()
    data = call_openai_json(client, MODEL_CLASSIFY, SYSTEM, USER, retries=1)
    comps: List[Dict[str, Any]] = []
    for x in (data.get("componentes") or []):
        if isinstance(x, str):
            x_clean = x.strip()
            found = next(
                (c for c in allowed if x_clean in (c["name"], c["code"])),
                None
            )
            if found and found not in comps:
                comps.append(found)
        if len(comps) >= max_k:
            break

    # Fallback heurístico
    if not comps:
        texto_ref = f"{ev_text or ''} {desempeno or ''}"
        comps = _heuristic_components(area, texto_ref, allowed)
    if not comps and allowed:
        comps = [allowed[0]]

    return comps

# =========================================================
# Rúbricas
# =========================================================
SPANISH_STOPWORDS = set("""
a al algo alguna algunas alguno algunos ante antes asi aun aunque bajo bien cada como con contra cual cuales cuando de del desde donde dos el ella ellas ello ellos en entre era erais eramos eran
eres es esa esas ese esos esta estaba estabais estabamos estaban estado estais estamos estan estar esto estos fue fui fuiste ha habia habian haber habiais habiamos haces hacen hacia han hasta hay
la las le les lo los mas me mi mia mias mio mios mientras mis mucha muchas mucho muchos muy nada ni no nos nosotras nosotros nuestra nuestras nuestro nuestros o os otra otras otro otros para pero
poco por porque pueda puede pueden pues que quien quienes se sea sean segun ser si siempre siendo sin sobre sois somos son soy su sus te tengo tiene tienen toda todas todo todos tu tus un una uno
unos usted ustedes vosotras vosotros y ya yo
""".split())

def _keywords(text: str, max_k: int = 12) -> List[str]:
    text = (text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation + "“”«»¡!¿?"))
    words = [w for w in re.findall(r"[a-záéíóúñü]{3,}", text) if w not in SPANISH_STOPWORDS]
    seen = set()
    out: List[str] = []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= max_k:
            break
    return out

def build_rubric_context(area: str, comps: list, ev_text: str, desempeno: str) -> dict:
    comps_context = [
        {
            "code": c.get("code",""),
            "name": c.get("name",""),
            "stage": c.get("stage","")
        }
        for c in (comps or [])
    ]
    kw = _keywords(f"{ev_text} {desempeno}", max_k=12)
    return {"area": area, "componentes": comps_context, "palabras_clave": kw}

def detect_language_for_text(area: str, ev_text: str, desempeno: str) -> str:
    """
    Devuelve 'en' o 'es' según el área y el texto.
    """
    area_up = (area or "").strip().upper()
    if area_up in ("INGLÉS", "INGLES", "ENGLISH"):
        return "en"

    text = f"{desempeno or ''} {ev_text or ''}".strip()
    if not text:
        return "es"
    text_low = text.lower()

    # Regla fuerte
    if re.match(r"^\s*to\s+[a-z]", text_low):
        return "en"

    # Tildes o ñ -> muy probable español
    if re.search(r"[áéíóúñü]", text_low):
        return "es"

    words = re.findall(r"[a-záéíóúñü]+", text_low)

    EN_COMMON = {
        "the", "and", "to", "of", "in", "for", "with", "from", "by", "on",
        "through", "explain", "analyze", "analyse", "describe", "political",
        "economic", "social", "cold", "war", "state"
    }
    ES_COMMON = {
        "el", "la", "los", "las", "y", "que", "de", "en", "por", "para",
        "con", "sobre", "político", "política", "económico", "económica",
        "social", "guerra", "fría"
    }

    en_score = sum(1 for w in words if w in EN_COMMON)
    es_score = sum(1 for w in words if w in ES_COMMON)

    if en_score > es_score:
        return "en"
    return "es"

def llm_generate_rubric(area: str, ev_text: str, desempeno: str, comps: list) -> dict:
    client = get_client()
    CONTEXTO = build_rubric_context(area, comps, ev_text, desempeno)
    lang = detect_language_for_text(area, ev_text, desempeno)

    if lang == "en":
        SYSTEM = (
            "You are an expert in curriculum assessment. You only return a valid JSON object. "
            "Generate OBSERVATIONAL rubrics, phrased positively, without numbers, percentages "
            "or expressions like 'at least'. "
            "Write all 'criterios', level descriptions and feedback comments in ENGLISH. "
            "Avoid visible lists or bullet points; use complete sentences."
        )
    else:
        SYSTEM = (
            "Eres experto en evaluación curricular. Devuelves SOLO un objeto JSON VÁLIDO. "
            "Genera rúbricas OBSERVACIONALES, redactadas en positivo, sin números, porcentajes "
            "ni expresiones como 'al menos'. "
            "Escribe todos los 'criterios', descripciones de niveles y comentarios de "
            "retroalimentación en ESPAÑOL, salvo términos propios del área. "
            "Evita listas o viñetas visibles: trabaja con oraciones completas."
        )

    instrucciones = [
        "Lee la Instancia verificadora y el Desempeño.",
        "Genera 2,3 o 4 'criterios' específicos y observables (solo textos breves).",
        "Para cada nivel ('bajo','minimo','medio','alto'), escribe una descripción en forma de una "
        "o varias oraciones que formen un único párrafo, positiva, sin cuantificar y sin saltos de línea.",
        "Describe de manera incremental lo que el estudiante demuestra en cada nivel.",
        "Además, genera para cada nivel ('bajo','minimo','medio','alto') un comentario de "
        "retroalimentación breve dirigido al estudiante, en segunda persona, que le indique "
        "cómo avanzar al siguiente nivel (o cómo sostener su desempeño en el nivel 'alto'), "
        "en un único párrafo sin saltos de línea.",
        "Para las descripciones de nivel, escribe un único párrafo coherente: usa conectores "
        "(por ejemplo 'además', 'también', 'por otro lado') y evita listar habilidades sueltas "
        "sin relación; todas las oraciones deben referirse al mismo foco de desempeño.",
    ]

    if lang == "en":
        instrucciones.append(
            "Write everything in ENGLISH (names of 'criterios', level descriptions and feedback comments)."
        )
    else:
        instrucciones.append(
            "Escribe todo en ESPAÑOL (nombres de 'criterios', descripciones de niveles y comentarios de retroalimentación)."
        )

    USER = "json\n" + json.dumps({
        "instrucciones": instrucciones,
        "contexto": CONTEXTO,
        "entrada": {"instancia_verificadora": ev_text, "desempeno": desempeno},
        "formato_esperado": {
            "criterios": ["...","...","..."],
            "niveles": {
                "bajo": ["..."],
                "minimo": ["..."],
                "medio": ["..."],
                "alto": ["..."]
            },
            "retroalimentacion": {
                "bajo": "...",
                "minimo": "...",
                "medio": "...",
                "alto": "..."
            }
        }
    }, ensure_ascii=False)

    resp = client.chat.completions.create(
        model=RUBRIC_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": USER}
        ],
        timeout=60.0,
    )
    content = _strip_code_fences(resp.choices[0].message.content or "{}")
    try:
        data = json.loads(content)
    except Exception:
        data = {}

    criterios = [str(x).strip() for x in (data.get("criterios") or []) if str(x).strip()]
    niveles = data.get("niveles") or {}
    retro   = data.get("retroalimentacion") or {}

    def _as_list(x):
        if isinstance(x, list):
            return [re.sub(r"\s+", " ", str(i).strip()) for i in x if str(i).strip()]
        if isinstance(x, str) and x.strip():
            parts = re.split(r"(?:\n|•|- )", x)
            return [re.sub(r"\s+", " ", s.strip()) for s in parts if s.strip()]
        return []

    def _as_str(x):
        if isinstance(x, list):
            joined = " ".join(str(i).strip() for i in x if str(i).strip())
            return re.sub(r"\s+", " ", joined).strip()
        if isinstance(x, str) and x.strip():
            return re.sub(r"\s+", " ", x.strip())
        return ""

    bajo   = _as_list(niveles.get("bajo"))
    minimo = _as_list(niveles.get("minimo"))
    medio  = _as_list(niveles.get("medio"))
    alto   = _as_list(niveles.get("alto"))

    fb_bajo   = _as_str(retro.get("bajo"))
    fb_minimo = _as_str(retro.get("minimo"))
    fb_medio  = _as_str(retro.get("medio"))
    fb_alto   = _as_str(retro.get("alto"))

    if not criterios:
        if lang == "en":
            criterios = [
                "Relevance and purpose",
                "Clarity and coherence",
                "Use of strategies and resources",
                "Adaptation to context"
            ]
        else:
            criterios = [
                "Pertinencia y propósito",
                "Claridad y coherencia",
                "Uso de estrategias y recursos",
                "Adecuación al contexto"
            ]

    fallback_es = [
        "Participa con apoyo.",
        "Sigue orientaciones básicas.",
        "Muestra avances consistentes."
    ]
    fallback_en = [
        "Participates with support.",
        "Follows basic guidance.",
        "Shows consistent progress."
    ]
    fallback = fallback_en if lang == "en" else fallback_es

    for lst in (bajo, minimo, medio, alto):
        if not lst:
            lst.extend(fallback)

    if not fb_bajo:
        fb_bajo = (
            "Take small steps and accept guidance so you can gradually incorporate what is still difficult for you."
            if lang == "en" else
            "Da pequeños pasos y acepta la guía para ir incorporando poco a poco lo que aún te cuesta."
        )
    if not fb_minimo:
        fb_minimo = (
            "Keep practicing consistently and look for opportunities to use what you already know in new situations."
            if lang == "en" else
            "Sigue practicando con constancia y busca oportunidades para usar lo que ya sabes en nuevas situaciones."
        )
    if not fb_medio:
        fb_medio = (
            "Maintain your level by practicing regularly and set yourself small challenges that demand a bit more from you."
            if lang == "en" else
            "Mantén tu nivel practicando de forma regular y proponte pequeños retos que te exijan un poco más."
        )
    if not fb_alto:
        fb_alto = (
            "Continue exploring ways to go deeper and share your strategies with others to consolidate your academic leadership."
            if lang == "en" else
            "Sigue explorando formas de profundizar y comparte tus estrategias con otros para consolidar y enriquecer tu evaluación."
        )

    return {
        "criterios": criterios[:4],
        "niveles": {
            "bajo": bajo,
            "minimo": minimo,
            "medio": medio,
            "alto": alto
        },
        "retroalimentacion": {
            "bajo": fb_bajo,
            "minimo": fb_minimo,
            "medio": fb_medio,
            "alto": fb_alto
        }
    }

def rubric_from_evidence(ev_text: str, desempeno: str, comps: list, area: str) -> dict:
    data = llm_generate_rubric(area, ev_text, desempeno, comps)
    lang = detect_language_for_text(area, ev_text, desempeno)

    def _textify_list(items) -> str:
        if not items:
            return ""
        sentences: List[str] = []
        for s in items:
            if not s:
                continue
            t = re.sub(r"\s+", " ", str(s).strip())
            if not t:
                continue
            if t[-1] not in ".?!…":
                t += "."
            sentences.append(t)
        paragraph = " ".join(sentences)
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        return paragraph

    def _as_str(x) -> str:
        if isinstance(x, list):
            joined = " ".join(str(i).strip() for i in x if str(i).strip())
            txt = re.sub(r"\s+", " ", joined).strip()
        elif isinstance(x, str):
            txt = re.sub(r"\s+", " ", x.strip())
        else:
            txt = ""
        if txt and not txt.endswith("."):
            txt += "."
        return txt

    niveles = data.get("niveles", {}) or {}
    retro   = data.get("retroalimentacion", {}) or {}

    descr_bajo   = _textify_list(niveles.get("bajo", []))
    descr_minimo = _textify_list(niveles.get("minimo", []))
    descr_medio  = _textify_list(niveles.get("medio", []))
    descr_alto   = _textify_list(niveles.get("alto", []))

    fb_bajo   = _as_str(retro.get("bajo", ""))
    fb_minimo = _as_str(retro.get("minimo", ""))
    fb_medio  = _as_str(retro.get("medio", ""))
    fb_alto   = _as_str(retro.get("alto", ""))

    if not descr_bajo:
        descr_bajo = (
            "The student shows an initial level of performance and needs frequent support to carry out the task."
            if lang == "en" else
            "El estudiante muestra un desempeño inicial y requiere apoyo frecuente para realizar la tarea."
        )
    if not descr_minimo:
        descr_minimo = (
            "The student carries out the task with occasional help and demonstrates basic progress in the performance."
            if lang == "en" else
            "El estudiante realiza la tarea con ayuda puntual y demuestra avances básicos en el desempeño."
        )
    if not descr_medio:
        descr_medio = (
            "The student carries out the task adequately and consistently in most situations."
            if lang == "en" else
            "El estudiante realiza la tarea de forma adecuada y consistente en la mayoría de las situaciones."
        )
    if not descr_alto:
        descr_alto = (
            "The student carries out the task with autonomy and flexibility and contributes more than expected for the level."
            if lang == "en" else
            "El estudiante realiza la tarea con autonomía, flexibilidad y aporta más de lo esperado para el nivel."
        )

    if not fb_bajo:
        fb_bajo = (
            "Keep practicing with your teacher’s support and use the suggestions to improve step by step."
            if lang == "en" else
            "Sigue practicando con apoyo del docente y acepta las sugerencias para mejorar poco a poco."
        )
    if not fb_minimo:
        fb_minimo = (
            "Reinforce what you already know by practicing in more situations and ask for help when you need it."
            if lang == "en" else
            "Refuerza lo que ya sabes practicando en más situaciones y pide ayuda cuando la necesites."
        )
    if not fb_medio:
        fb_medio = (
            "Maintain your effort and look for small new challenges to continue improving your performance."
            if lang == "en" else
            "Mantén tu esfuerzo y busca pequeños retos nuevos para seguir mejorando tu desempeño."
        )
    if not fb_alto:
        fb_alto = (
            "Keep sharing your strategies with others and set yourself challenges that allow you to keep growing."
            if lang == "en" else
            "Sigue compartiendo tus estrategias con otros y ponte desafíos que te permitan seguir creciendo."
        )

    return {
        "criterios": data.get("criterios", []),
        "descripcion_bajo":   descr_bajo,
        "descripcion_minimo": descr_minimo,
        "descripcion_medio":  descr_medio,
        "descripcion_alto":   descr_alto,
        "comentario_bajo":    fb_bajo,
        "comentario_minimo":  fb_minimo,
        "comentario_medio":   fb_medio,
        "comentario_alto":    fb_alto,
    }

# =========================================================
# Excel (utilidad opcional, no usada por Streamlit)
# =========================================================
def append_to_excel(output_path: str, df_new: pd.DataFrame, sheet_name: str = "Instancias") -> int:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            df_old = pd.read_excel(output_path, sheet_name=sheet_name, engine="openpyxl")
        except Exception:
            df_old = pd.DataFrame(columns=df_new.columns)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        subset_cols = [c for c in ["Desempeño_id", "Instancia Verificadora"] if c in df_all.columns]
        if subset_cols:
            df_all = df_all.drop_duplicates(subset=subset_cols, keep="first")
        else:
            df_all = df_all.drop_duplicates(keep="first")
    else:
        df_all = df_new.copy()

    if "Ítem" in df_all.columns:
        df_all = df_all.drop(columns=["Ítem"])
    df_all.insert(0, "Ítem", range(1, len(df_all) + 1))

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        df_all.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.sheets[sheet_name]
        ws.freeze_panes(1, 0)
        last_row = len(df_all)
        last_col = len(df_all.columns) - 1
        ws.autofilter(0, 0, last_row, last_col)
        widths = {
            "Ítem": 6, "Área": 14, "Grado": 14, "Unidad": 8, "Trimestre": 12, "Desempeño_id": 30,
            "Título Desempeño": 36,
            "Desempeño Precisado": 60,
            "Desempeño Precisado breve": 40,
            "Instancia Verificadora": 60,
            "Componentes (nombres)": 40, "Componentes (códigos)": 20, "Componentes (etapas)": 24,
            "Nivel Bajo": 60, "Nivel Mínimo": 60, "Nivel Medio": 60, "Nivel Alto": 60,
            "Comentario Bajo": 60, "Comentario Mínimo": 60, "Comentario Medio": 60, "Comentario Alto": 60,
        }
        for i, col in enumerate(df_all.columns):
            ws.set_column(i, i, widths.get(col, 18))
    return len(df_all)

# =========================================================
# Construcción de filas
# =========================================================
def build_rows_for_unit(
    unit_num: int,
    area: str,
    grado_num: int,
    area_inits: str,
    desempenos: List[Dict[str,Any]],
    rows_sink: List[Dict[str,Any]],
    rubric_workers: int = 6
):

    pairs: List[Tuple[int, Dict[str,Any], int, str]] = []
    for d_idx, d in enumerate(desempenos, start=1):
        for e_idx, iv in enumerate(d.get("instancias", [])[:], start=1):
            pairs.append((d_idx, d, e_idx, iv))

    comps_por_des: Dict[int, List[Dict[str,Any]]] = {}
    for d_idx, d in enumerate(desempenos, start=1):
        texto_dp = d.get("desempeno_precisado","") or d.get("desempeño_precisado","") or ""
        comps_por_des[d_idx] = llm_pick_components_for_instance(
            area,
            grado_num,
            texto_dp,
            texto_dp,
            max_k=3
        )

    def build_row(d_idx: int, d: Dict[str,Any], e_idx: int, iv_text: str) -> Dict[str,Any]:
        comps = comps_por_des.get(d_idx, [])
        dp_full = d.get("desempeno_precisado","") or d.get("desempeño_precisado","") or ""
        titulo_dp = make_desempeno_title(dp_full, d.get("titulo","") or "")
        dp_breve = make_desempeno_short(dp_full)

        rb = rubric_from_evidence(iv_text, dp_full, comps, area)
        desempeno_id = f"{area_inits}{grado_num}U{unit_num}_{(d.get('codigo') or f'D{d_idx}').strip()}_IV{e_idx}"

        return {
            "Área": area,
            "Grado": f"{grado_num} - {grade_name(grado_num)}",
            "Unidad": f"U{unit_num}",
            "Trimestre": f"Trimestre {1 if unit_num in (1,2) else 2 if unit_num in (3,4) else 3}",
            "Desempeño_id": desempeno_id,
            "Título Desempeño": titulo_dp,
            "Desempeño Precisado": dp_full,
            "Desempeño Precisado breve": dp_breve,
            "Instancia Verificadora": iv_text,
            "Componentes (nombres)": ", ".join([c["name"] for c in comps]) if comps else "",
            "Componentes (códigos)": ", ".join([c["code"] for c in comps]) if comps else "",
            "Componentes (etapas)": ", ".join([c.get("stage_label") or grade_name(grado_num) for c in comps]) if comps else "",
            "Nivel Bajo":   rb.get("descripcion_bajo",""),
            "Nivel Mínimo": rb.get("descripcion_minimo",""),
            "Nivel Medio":  rb.get("descripcion_medio",""),
            "Nivel Alto":   rb.get("descripcion_alto",""),
            "Comentario Bajo":   rb.get("comentario_bajo",""),
            "Comentario Mínimo": rb.get("comentario_minimo",""),
            "Comentario Medio":  rb.get("comentario_medio",""),
            "Comentario Alto":   rb.get("comentario_alto",""),
        }

    if rubric_workers and rubric_workers > 1:
        with ThreadPoolExecutor(max_workers=int(rubric_workers)) as ex:
            futures = [
                ex.submit(build_row, d_idx, d, e_idx, iv_text)
                for d_idx, d, e_idx, iv_text in pairs
            ]
            for fut in as_completed(futures):
                try:
                    rows_sink.append(fut.result())
                except Exception as e:
                    print("⚠️ Error en rúbrica:", e)
    else:
        for d_idx, d, e_idx, iv_text in pairs:
            try:
                rows_sink.append(build_row(d_idx, d, e_idx, iv_text))
            except Exception as e:
                print("⚠️ Error en rúbrica:", e)

# =========================================================
# Parámetros globales por defecto
# =========================================================
ALWAYS_UNITS = [1, 2, 3, 4, 5, 6]
PARSER_IVS_PER_DES_INITIAL = 4
RUBRIC_WORKERS = 6

# =========================================================
# FUNCIÓN PRINCIPAL PARA LA APP
# =========================================================
def run_pipeline(
    raw_text: str,
    area: str,
    grado_num: int,
    ivs_per_desempeno: int
) -> pd.DataFrame:
    """
    Ejecuta todo el pipeline sobre el texto bruto (raw_text) para
    un área y grado dados, generando una cantidad de instancias por desempeño
    controlada por el usuario (ivs_per_desempeno).
    Devuelve un DataFrame con todas las filas.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("El texto de entrada está vacío.")

    if area not in COMPONENTS_CATALOG:
        raise ValueError(f"Área desconocida: {area}")

    if not isinstance(grado_num, int) or grado_num < -2 or grado_num > 11:
        raise ValueError("GRADO_NUM fuera de rango (-2 a 11).")

    if ivs_per_desempeno <= 0:
        raise ValueError("La cantidad de instancias por desempeño debe ser mayor que 0.")

    # 1) Partir en unidades
    units_list = split_unidades_dispatch(raw_text)
    units_by_u: Dict[int, str] = {}
    for u, ch in units_list:
        if isinstance(u, int):
            if u not in units_by_u or len(ch) > len(units_by_u[u]):
                units_by_u[u] = ch

    # 2) Asegurar U1..U6 (ajustada a tu estructura original)
    for u in ALWAYS_UNITS:
        units_by_u.setdefault(u, "")

    # 3) Parsear cada unidad y ajustar IV según la cantidad elegida por el usuario
    parsed_by_unit: Dict[int, Dict[str, Any]] = {}
    for u in sorted(units_by_u.keys()):
        parsed = parse_unidad_text(
            units_by_u[u],
            n_instancias_exact=int(ivs_per_desempeno or PARSER_IVS_PER_DES_INITIAL)
        )
        parsed["desempenos"] = enforce_iv_targets_by_unit(
            parsed.get("desempenos", []),
            ivs_per_des_user=int(ivs_per_desempeno)
        )
        parsed_by_unit[u] = parsed

    # 4) Si alguna unidad queda sin desempeños, replicar desde la unidad más cercana (pares con pares)
    def replicate_desempenos_from_nearest(unit: int) -> List[Dict[str,Any]]:
        candidates = sorted(parsed_by_unit.keys())
        preferred: List[int] = []
        for delta in range(1, 7):
            for sign in (+1, -1):
                u2 = unit + sign*delta
                if u2 in parsed_by_unit and parsed_by_unit[u2].get("desempenos"):
                    if (u2 % 2) == (unit % 2):
                        preferred.append(u2)
        if not preferred:
            for u2 in candidates:
                if parsed_by_unit[u2].get("desempenos"):
                    preferred.append(u2)
        if not preferred:
            return []
        donor = preferred[0]
        return copy.deepcopy(parsed_by_unit[donor]["desempenos"])

    for u in ALWAYS_UNITS:
        if not parsed_by_unit.get(u, {}).get("desempenos"):
            parsed_by_unit[u] = {
                "unidad": u,
                "desempenos": replicate_desempenos_from_nearest(u)
            }

    # 5) Construir filas
    area_inits = initials(area)
    rows_all: List[Dict[str, Any]] = []

    for u in ALWAYS_UNITS:
        des = parsed_by_unit.get(u, {}).get("desempenos", [])
        if not des:
            continue
        build_rows_for_unit(
            u,
            area,
            grado_num,
            area_inits,
            des,
            rows_all,
            rubric_workers=RUBRIC_WORKERS
        )

    if not rows_all:
        raise RuntimeError("No se generaron filas. Revisa el texto de entrada o la detección de unidades.")

    df = pd.DataFrame(rows_all)

    order_cols = [
        "Ítem",
        "Área","Grado","Unidad","Trimestre","Desempeño_id",
        "Título Desempeño","Desempeño Precisado","Desempeño Precisado breve","Instancia Verificadora",
        "Componentes (nombres)","Componentes (códigos)","Componentes (etapas)",
        "Nivel Bajo","Nivel Mínimo","Nivel Medio","Nivel Alto",
        "Comentario Bajo","Comentario Mínimo","Comentario Medio","Comentario Alto",
    ]
    df = df.reindex(columns=[c for c in order_cols if c in df.columns])

    if "Ítem" in df.columns:
        df = df.drop(columns=["Ítem"])
    df.insert(0, "Ítem", range(1, len(df) + 1))

    return df
