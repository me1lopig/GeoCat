import streamlit as st
import requests
import numpy as np
import re
from difflib import get_close_matches, SequenceMatcher
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import geopandas as gpd
import pandas as pd
import tempfile, os, zipfile

try:
    import pytesseract
    TESSERACT_OK = True
except ImportError:
    TESSERACT_OK = False

st.set_page_config(page_title="Visor Geologia ICGC", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
WMS_BASE     = "https://geoserveis.icgc.cat/servei/catalunya/geologia-territorial/wms"
CAPA_WMS     = "unitats-geologiques-50000"
ARCGIS_BASE  = "https://maps.icgc.cat/vector01/rest/services/geologia_territorial/MapServer/38/query"
ARCGIS_LAYER = "https://maps.icgc.cat/vector01/rest/services/geologia_territorial/MapServer/38"
CAMP_CODI    = "Codi"
EPSG_ICGC    = 25831

# Servicios de fondo disponibles (URL, capa, versión WMS, parámetro CRS)
FONS_SERVEIS = {
    "Ortofoto color (ICGC)": {
        "url":    "https://geoserveis.icgc.cat/servei/catalunya/orto-territorial/wms",
        "layer":  "ortofoto_color_serie_anual",
        "ver":    "1.3.0", "srs_key": "CRS", "fmt": "image/jpeg",
    },
    "Mapa topográfico (ICGC)": {
        "url":    "https://geoserveis.icgc.cat/servei/catalunya/mapa-base/wms",
        "layer":  "topografic",
        "ver":    "1.1.1", "srs_key": "SRS", "fmt": "image/png",
    },
    "Mapa gris topográfico (ICGC)": {
        "url":    "https://geoserveis.icgc.cat/icc_fonstopografic/wms/service",
        "layer":  "mtc25msg",
        "ver":    "1.1.1", "srs_key": "SRS", "fmt": "image/jpeg",
    },
    "Sin fondo (solo geología)": None,
}

# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONS DE SERVEI
# ──────────────────────────────────────────────────────────────────────────────

def bbox_utm_25831(x_centre, y_centre, ample_m, alt_m):
    mx, my = ample_m / 2.0, alt_m / 2.0
    return (x_centre - mx, y_centre - my, x_centre + mx, y_centre + my)


def calcular_px(ample_m, alt_m, max_dim=1024):
    if ample_m >= alt_m:
        w = int(max_dim); h = max(1, int(round(max_dim * alt_m / ample_m)))
    else:
        h = int(max_dim); w = max(1, int(round(max_dim * ample_m / alt_m)))
    return w, h


def obtenir_imatge_wms(minx, miny, maxx, maxy, ample_px, alt_px):
    params = {
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
        "LAYERS": CAPA_WMS, "STYLES": "", "FORMAT": "image/png",
        "TRANSPARENT": "false", "CRS": f"EPSG:{EPSG_ICGC}",
        "BBOX": f"{minx},{miny},{maxx},{maxy}",
        "WIDTH": int(ample_px), "HEIGHT": int(alt_px),
    }
    resp = requests.get(WMS_BASE, params=params, timeout=60)
    resp.raise_for_status()
    ct = resp.headers.get("Content-Type", "")
    if "image" not in ct:
        raise RuntimeError(f"WMS no ha retornat imatge. CT:{ct}\n{resp.text[:300]}")
    return resp.content



def obtenir_imatge_fons(minx, miny, maxx, maxy, ample_px, alt_px, servei_key):
    """
    Descarga la imagen de fondo (ortofoto o cartografía) del servicio seleccionado.
    Retorna bytes PNG/JPEG o None si el servicio no está disponible.
    """
    servei = FONS_SERVEIS.get(servei_key)
    if servei is None:
        return None
    params = {
        "SERVICE":     "WMS",
        "VERSION":     servei["ver"],
        "REQUEST":     "GetMap",
        "LAYERS":      servei["layer"],
        "STYLES":      "",
        "FORMAT":      servei["fmt"],
        "TRANSPARENT": "false",
        servei["srs_key"]: f"EPSG:{EPSG_ICGC}",
        "BBOX":        f"{minx},{miny},{maxx},{maxy}",
        "WIDTH":       int(ample_px),
        "HEIGHT":      int(alt_px),
    }
    try:
        resp = requests.get(servei["url"], params=params, timeout=30)
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "")
        if "image" not in ct:
            return None
        return resp.content
    except Exception:
        return None


def compondre_imatges(fons_bytes, geol_bytes, opacitat=0.6):
    """
    Combina la imagen de fondo con la imagen geológica aplicando una opacidad
    a la capa geológica (0.0 = transparente, 1.0 = opaco).
    Retorna bytes PNG de la imagen compuesta.
    """
    fons = Image.open(BytesIO(fons_bytes)).convert("RGBA")
    geol = Image.open(BytesIO(geol_bytes)).convert("RGBA")

    # Adaptar mida si cal (el fons pot tenir mides lleugerament diferents)
    if fons.size != geol.size:
        geol = geol.resize(fons.size, Image.LANCZOS)

    # Aplicar opacitat al canal alfa de la geologia
    r, g, b, a = geol.split()
    # Píxeles blancos (fondo neutro del WMS geología) → transparentes
    arr_r = np.array(r)
    arr_g = np.array(g)
    arr_b = np.array(b)
    # Máscara: píxeles casi blancos del WMS geología → alpha=0 (transparente)
    blanc_mask = (arr_r > 248) & (arr_g > 248) & (arr_b > 248)
    arr_a = np.array(a).astype(np.float32)
    arr_a[blanc_mask] = 0
    arr_a[~blanc_mask] = arr_a[~blanc_mask] * opacitat
    a_nou = Image.fromarray(arr_a.astype(np.uint8))
    geol_mod = Image.merge("RGBA", (r, g, b, a_nou))

    # Composició
    resultat = fons.copy()
    resultat.paste(geol_mod, (0, 0), geol_mod)

    buf = BytesIO()
    resultat.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    return buf.read()



def obtenir_unitats_bbox(minx, miny, maxx, maxy):
    """
    Consulta directa al ArcGIS REST de la capa 38 (unitats-geologiques-50000)
    con el BBOX del área solicitada.

    Parámetros clave:
    - returnGeometry=false  → solo atributos, sin polígonos (muy rápido)
    - returnDistinctValues=true → una fila por Codi único (sin duplicados)
    - orderByFields=Ordre ASC   → ordenado cronoestratigráficamente

    Retorna: lista de dicts con Codi, Descripcio, Ordre, Era, Periode, Epoca
             o None si el servicio no responde.
    """
    params = {
        "f":                    "json",
        "geometry":             f"{minx},{miny},{maxx},{maxy}",
        "geometryType":         "esriGeometryEnvelope",
        "inSR":                 EPSG_ICGC,
        "spatialRel":           "esriSpatialRelIntersects",
        "outFields":            "Codi,Descripcio,Ordre,Eo,Era,Periode,Epoca",
        "returnGeometry":       "false",
        "returnDistinctValues": "true",
        "orderByFields":        "Ordre ASC",
    }
    try:
        resp = requests.get(ARCGIS_BASE, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return None
        features = data.get("features", [])
        if not features:
            return None
        return [f.get("attributes", {}) for f in features]
    except Exception:
        return None


def obtenir_renderer():
    """
    Descarga el renderer UniqueValue (colores oficiales por Codi) de la capa ArcGIS.
    Retorna dict: Codi → {"color": (R,G,B), "ordre": int}
    """
    try:
        resp = requests.get(ARCGIS_LAYER + "?f=pjson", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        infos = data.get("drawingInfo", {}).get("renderer", {}).get("uniqueValueInfos", [])
        renderer = {}
        for info in infos:
            val   = info.get("value", "")
            color = info.get("symbol", {}).get("color", [180, 180, 180, 255])
            parts = val.split(",")
            if len(parts) >= 2:
                try:    ordre = int(parts[0].strip())
                except: ordre = 999
                codi = parts[1].strip()
            else:
                codi, ordre = val.strip(), 999
            if codi:
                renderer[codi] = {"color": tuple(color[:3]), "ordre": ordre}
        return renderer
    except Exception as e:
        st.warning(f"No se ha podido descargar el renderer ({e}). Se usarán colores neutros.")
        return {}


def obtenir_descripcions(codis):
    """Consulta ArcGIS REST para obtener Descripcio, Era, Periode, Epoca sin geometría."""
    if not codis:
        return {}
    codi_list = ",".join(f"'{c}'" for c in codis)
    params = {
        "f": "json", "where": f"Codi IN ({codi_list})",
        "outFields": "Codi,Descripcio,Ordre,Eo,Era,Periode,Epoca",
        "returnGeometry": "false", "returnDistinctValues": "true",
    }
    try:
        resp = requests.get(ARCGIS_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return {}
        return {
            feat["attributes"]["Codi"]: feat["attributes"]
            for feat in data.get("features", [])
            if feat.get("attributes", {}).get("Codi")
        }
    except Exception:
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# DETECCIÓ DE CODIS — MÈTODE 1: OCR (principal)
# ──────────────────────────────────────────────────────────────────────────────

def _match_token(token, codis_set, cutoff=0.72):
    """
    Matching robusto de un token OCR contra la lista de codis del renderer.
    Orden de prioridad:
    1. Exacto (case-sensitive)
    2. Case-insensitive
    3. Sustitución de confusiones OCR habituales: O↔Q, l↔1, 0↔O, |→l
    4. Fuzzy con bonus si el primer carácter coincide
    """
    if not token:
        return None
    # 1. Exacte
    if token in codis_set:
        return token
    # 2. Case-insensitive
    m = next((c for c in codis_set if c.lower() == token.lower()), None)
    if m:
        return m
    # 3. Sustituciones de OCR
    subs = {'O': 'Q', 'Q': 'O', 'l': '1', '0': 'O', '|': 'l'}
    if token[0] in subs:
        tc = subs[token[0]] + token[1:]
        if tc in codis_set:
            return tc
        m = next((c for c in codis_set if c.lower() == tc.lower()), None)
        if m:
            return m
    # 4. Fuzzy
    candidates = get_close_matches(token, list(codis_set), n=3, cutoff=cutoff)
    if candidates:
        best, best_sc = None, 0.0
        for cand in candidates:
            sc = SequenceMatcher(None, token, cand).ratio()
            if token[0].lower() == cand[0].lower():
                sc *= 1.1  # bonus primer caràcter igual
            if sc > best_sc:
                best, best_sc = cand, sc
        if best and best_sc >= cutoff:
            return best
    return None


def detectar_codis_ocr(img_bytes, codis_renderer):
    """
    Extrae los codis geológicos de las etiquetas de texto del mapa con OCR multi-umbral.

    Pipeline:
    1. Aplicar Tesseract con DOS umbrales de binarización:
       - Umbral 140: captura texto oscuro sobre fondo claro Y texto en unidades de color oscuro
                     (SDc marrón, Caps verde, Sf azul, NMs amarillo)
       - Umbral 170: captura texto estándar sobre fondo pastel claro (Qbcn, Qpa, Qp)
    2. Por cada token, aplicar _match_token (exacto → CI → sust. OCR → fuzzy)
    3. Validar: aceptar codi si aparece en 2+ umbrales O tiene confianza > 60
       → elimina falsos positivos que aparecen en un solo umbral con baja confianza

    Retorna: dict Codi → max_confianza
    """
    if not TESSERACT_OK:
        return {}

    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    gray_arr = np.array(img.convert("L"))
    cfg = r'--psm 11 --oem 3'
    codis_set = set(codis_renderer)

    # Codi → {"umbrals": set, "max_conf": int}
    codi_info = {}

    for umbral in [140, 170]:
        bin_arr = np.where(gray_arr < umbral, 0, 255).astype(np.uint8)
        img_bin = Image.fromarray(bin_arr).resize(
            (img.width * 3, img.height * 3), Image.LANCZOS
        )
        try:
            data = pytesseract.image_to_data(
                img_bin, config=cfg, output_type=pytesseract.Output.DICT
            )
        except Exception:
            continue

        for txt, conf in zip(data['text'], data['conf']):
            txt = txt.strip()
            if not txt or conf <= 25 or not (2 <= len(txt) <= 12):
                continue
            codi = _match_token(txt, codis_set)
            if codi:
                if codi not in codi_info:
                    codi_info[codi] = {"umbrals": set(), "max_conf": 0}
                codi_info[codi]["umbrals"].add(umbral)
                codi_info[codi]["max_conf"] = max(codi_info[codi]["max_conf"], int(conf))

    # Validar: 2+ umbrals O confiança > 60
    return {
        codi: info["max_conf"]
        for codi, info in codi_info.items()
        if len(info["umbrals"]) >= 2 or info["max_conf"] > 60
    }


# ──────────────────────────────────────────────────────────────────────────────
# DETECCIÓ DE CODIS — MÈTODE 2: COLORS (fallback i complement)
# ──────────────────────────────────────────────────────────────────────────────

def detectar_codis_colors(img_bytes, renderer, min_fraccio=0.0003, cubs_px=6):
    """
    Detecta codis a partir de los colores dominantes de la imagen.

    1. Cuantiza los píxeles no-blancos en cubos RGB de cubs_px
    2. Selecciona clusters con cobertura > min_fraccio
    3. Asigna cada cluster al Codi del renderer con distancia euclidiana mínima

    Se usa como complemento del OCR para capturar unidades grandes sin etiqueta visible.
    """
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    pixels = np.array(img).reshape(-1, 3).astype(np.float32)

    mask = ~((pixels[:,0] > 240) & (pixels[:,1] > 240) & (pixels[:,2] > 240))
    pf = pixels[mask]
    if len(pf) == 0:
        return {}
    total = len(pf)

    quant = (pf // cubs_px).astype(np.int32)
    unique, counts = np.unique(quant, axis=0, return_counts=True)
    idx_ord = np.argsort(-counts)

    dominant = []
    for i in idx_ord:
        if counts[i] / total < min_fraccio:
            break
        r, g, b = [int(x) for x in unique[i] * cubs_px + cubs_px // 2]
        dominant.append((r, g, b, int(counts[i])))

    if not dominant:
        return {}

    renderer_items = list(renderer.items())
    codi_px = {}
    for r0, g0, b0, npx in dominant:
        best_codi, best_dist = None, 9999.0
        for codi, info in renderer_items:
            r1, g1, b1 = info["color"]
            d = ((r0-r1)**2 + (g0-g1)**2 + (b0-b1)**2) ** 0.5
            if d < best_dist:
                best_dist, best_codi = d, codi
        if best_codi:
            codi_px[best_codi] = codi_px.get(best_codi, 0) + npx

    return codi_px


# ──────────────────────────────────────────────────────────────────────────────
# FUSIÓN: OCR (principal) + COLORES (complemento para zonas grandes sin etiqueta)
# ──────────────────────────────────────────────────────────────────────────────

def fusionar_deteccions(ocr_codis, color_codis, total_px, min_pct_color=0.5):
    """
    Combina los resultados de OCR y colores siguiendo esta lógica:
    - Todos los codis del OCR se aceptan (alta precisión).
    - Los codis SOLO de colores se aceptan si cubren > min_pct_color% de la imagen
      (zona grande → seguramente real, no ruido).
    - Los codis SOLO de colores con cobertura pequeña se descartan (probable falso positivo).

    Retorna: set de codis finales
    """
    finals = set(ocr_codis.keys())

    for codi, npx in color_codis.items():
        if codi in finals:
            continue  # ya cubierto por OCR
        pct = npx / total_px * 100
        if pct >= min_pct_color:
            finals.add(codi)

    return finals


# ──────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓ DE LA LLEGENDA
# ──────────────────────────────────────────────────────────────────────────────

def construir_llegenda(codis_finals, renderer, descripcions, output_path):
    """
    Genera una imatge PNG de llegenda amb:
    - Rectangle de color real (del renderer ArcGIS)
    - Codi geològic
    - Descripció litològica
    - Edat (Era + Periode)
    Ordenat per ordre cronoestratigràfic del renderer.
    """
    if not codis_finals:
        return None

    codis_ordenats = sorted(
        codis_finals,
        key=lambda c: renderer.get(c, {}).get("ordre", 999)
    )

    # Fonts
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    try:
        font_title = ImageFont.truetype(font_paths[0], 18)
        font_body  = ImageFont.truetype(font_paths[1], 16)
        font_small = ImageFont.truetype(font_paths[1], 14)
    except Exception:
        try:
            font_title = ImageFont.load_default(size=18)
            font_body  = ImageFont.load_default(size=16)
            font_small = ImageFont.load_default(size=14)
        except Exception:
            font_title = font_body = font_small = ImageFont.load_default()

    pad, sym_w, sym_h, gap = 18, 40, 24, 12
    title = "Leyenda — Unidades geológicas presentes en el área"

    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    def tw(text, font):
        bb = dummy.textbbox((0, 0), text, font=font)
        return bb[2] - bb[0]

    # Calcular amplada màxima necessària
    max_w = tw(title, font_title)
    for codi in codis_ordenats:
        info = descripcions.get(codi, {})
        desc = str(info.get("Descripcio", "")) if info else ""
        era  = str(info.get("Era",     ""))    if info else ""
        per  = str(info.get("Periode", ""))    if info else ""
        for txt, font in [(codi, font_body), (desc, font_small), (era + (f", {per}" if per not in ("","None","nan") else ""), font_small)]:
            if txt and txt not in ("None","nan"):
                max_w = max(max_w, tw(txt, font))

    # Calcular alçada total
    row_heights = []
    for codi in codis_ordenats:
        info  = descripcions.get(codi, {})
        desc  = str(info.get("Descripcio","")) if info else ""
        era   = str(info.get("Era",""))        if info else ""
        n = 1
        if desc not in ("","None","nan"): n += 1
        if era  not in ("","None","nan"): n += 1
        row_heights.append(max(sym_h + 8, n * 19 + 8))

    img_w = pad * 2 + sym_w + gap + max_w + pad
    img_h = pad + 26 + pad + sum(rh + 8 for rh in row_heights) + pad

    img  = Image.new("RGB", (int(img_w), int(img_h)), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((pad, pad), title, fill=(20, 20, 20), font=font_title)

    y = pad + 26 + pad
    for i, codi in enumerate(codis_ordenats):
        rh   = row_heights[i]
        info = descripcions.get(codi, {})
        desc = str(info.get("Descripcio","")) if info else ""
        era  = str(info.get("Era",""))        if info else ""
        per  = str(info.get("Periode",""))    if info else ""

        color = renderer.get(codi, {}).get("color", (190, 190, 190))
        draw.rectangle(
            [pad, int(y + rh/2 - sym_h/2), pad + sym_w, int(y + rh/2 + sym_h/2)],
            fill=color, outline=(80, 80, 80)
        )

        tx, ty = pad + sym_w + gap, int(y)
        draw.text((tx, ty), codi, fill=(20, 20, 20), font=font_body)
        ty += 20
        if desc not in ("","None","nan"):
            draw.text((tx, ty), desc, fill=(50, 50, 50), font=font_small)
            ty += 17
        age = era if era not in ("","None","nan") else ""
        if per not in ("","None","nan") and age:
            age += f", {per}"
        elif per not in ("","None","nan"):
            age = per
        if age:
            draw.text((tx, ty), age, fill=(110, 110, 110), font=font_small)

        y += rh + 8

    img.save(output_path)
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# INTERFÍCIE STREAMLIT
# ──────────────────────────────────────────────────────────────────────────────

st.title("🗺️ Generador de Cartografía Geológica (ICGC)")

metode_info = "OCR (Tesseract) + colores dominantes" if TESSERACT_OK else "colores dominantes (Tesseract no instalado)"
st.markdown(f"""
Consulta el **WMS Geología Territorial 1:50.000 del ICGC** y genera:

1. **Mapa geológico** PNG del área seleccionada.
2. **Leyenda específica** construida identificando las unidades presentes:
   - **Método principal**: OCR sobre las etiquetas de texto del mapa *(alta precisión)*.
   - **Complemento**: análisis de colores dominantes *(captura zonas grandes sin etiqueta)*.
   - Método activo: `{metode_info}`
3. Archivos **CSV** y **GeoJSON** si el servicio vectorial es accesible.

> Coordenadas **ETRS89 UTM zona 31N (EPSG:25831)**
""")

if not TESSERACT_OK:
    st.warning("⚠️ `pytesseract` no instalado. Se usará solo el análisis de colores. "
               "Para activar OCR: `pip install pytesseract` + `apt install tesseract-ocr`")

with st.sidebar:
    st.header("Parámetros de entrada")
    st.info("Coordenadas ETRS89 UTM zona 31N (EPSG:25831)")
    st.subheader("Centro del rectángulo (UTM)")
    c1, c2 = st.columns(2)
    x_centre = c1.number_input("X (m)", value=431_200.0, format="%.1f")
    y_centre = c2.number_input("Y (m)", value=4_582_200.0, format="%.1f")
    st.subheader("Dimensiones del rectángulo")
    d1, d2 = st.columns(2)
    ample = d1.number_input("Ancho (m)", value=5_000.0, min_value=100.0)
    alt   = d2.number_input("Alto (m)",   value=4_000.0, min_value=100.0)
    st.subheader("Visualización")
    fons_key = st.selectbox(
        "Capa de fondo",
        options=list(FONS_SERVEIS.keys()),
        index=0,
    )
    opacitat = st.slider(
        "Opacidad geología (%)",
        min_value=10, max_value=100, value=70, step=5,
        help="100% = geología opaca sin fondo visible"
    )
    executar = st.button("🚀 Generar mapa y leyenda", type="primary")

if executar:
    with st.spinner("Conectando con ICGC…"):
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                minx, miny, maxx, maxy = bbox_utm_25831(x_centre, y_centre, ample, alt)
                w_px, h_px = calcular_px(ample, alt)
                total_px = w_px * h_px

                # 1. Imatge WMS ────────────────────────────────────────────────
                with st.spinner("Descargando mapa WMS…"):
                    img_bytes = obtenir_imatge_wms(minx, miny, maxx, maxy, w_px, h_px)
                img_path = os.path.join(tmpdir, "icgc_mapa.png")
                with open(img_path, "wb") as f:
                    f.write(img_bytes)

                # 1b. Imatge de fons i composició ─────────────────────────────
                fons_bytes = None
                if fons_key != "Sin fondo (solo geología)":
                    with st.spinner(f"Descargando capa de fondo: {fons_key}…"):
                        fons_bytes = obtenir_imatge_fons(
                            minx, miny, maxx, maxy, w_px, h_px, fons_key
                        )
                    if fons_bytes is None:
                        st.warning(f"⚠️ No se ha podido descargar la capa de fondo '{fons_key}'. "
                                   "Se mostrará solo la geología.")

                # Componer geología + fondo si tenemos los dos
                if fons_bytes is not None:
                    with st.spinner("Componiendo geología + fondo…"):
                        img_comp = compondre_imatges(fons_bytes, img_bytes, opacitat/100)
                    img_path_show = os.path.join(tmpdir, "icgc_compost.png")
                    with open(img_path_show, "wb") as f:
                        f.write(img_comp)
                else:
                    img_path_show = img_path

                # 2. Renderer de colors ────────────────────────────────────────
                with st.spinner("Descargando renderer de colores…"):
                    renderer = obtenir_renderer()

                codis_renderer = set(renderer.keys())

                # 3. Identificación de unidades — método prioritario: API directa ──
                unitats_api  = None
                ocr_codis    = {}
                color_codis  = {}
                metode_used  = ""

                with st.spinner("Consultando unidades geológicas del área (API)…"):
                    unitats_api = obtenir_unitats_bbox(minx, miny, maxx, maxy)

                if unitats_api is not None:
                    # ── MÉTODO PRINCIPAL: API directa ─────────────────────────
                    # Resultado exacto al 100%: el servidor devuelve exactamente
                    # los codis que intersectan con el BBOX, sin OCR ni colores.
                    metode_used  = "API directa (ArcGIS REST)"
                    codis_finals = {u["Codi"] for u in unitats_api if u.get("Codi")}
                    descripcions = {
                        u["Codi"]: u for u in unitats_api if u.get("Codi")
                    }
                else:
                    # ── FALLBACK: OCR + colores ───────────────────────────────
                    # Solo si el servicio ArcGIS REST no responde
                    st.warning("⚠️ API directa no disponible. Usando OCR + análisis de colores.")
                    metode_used = "OCR + colores dominantes (fallback)"

                    if TESSERACT_OK:
                        with st.spinner("Aplicando OCR sobre las etiquetas del mapa…"):
                            ocr_codis = detectar_codis_ocr(img_bytes, codis_renderer)
                    with st.spinner("Analizando colores dominantes de la imagen…"):
                        color_codis = detectar_codis_colors(img_bytes, renderer)
                    codis_finals = fusionar_deteccions(ocr_codis, color_codis, total_px)
                    if codis_finals:
                        with st.spinner(f"Obteniendo descripciones de {len(codis_finals)} unidades…"):
                            descripcions = obtenir_descripcions(list(codis_finals))
                    else:
                        descripcions = {}

                # 5. Llegenda ─────────────────────────────────────────────────
                llegenda_path = os.path.join(tmpdir, "icgc_llegenda.png")
                te_llegenda = False
                if codis_finals:
                    r = construir_llegenda(codis_finals, renderer, descripcions, llegenda_path)
                    te_llegenda = (r is not None)

                # 6. Mensaje de éxito ───────────────────────────────────────────
                if codis_finals:
                    st.success(
                        f"✅ **{len(codis_finals)} unidades geológicas identificadas** — "
                        f"Método: *{metode_used}*"
                    )
                else:
                    st.warning("⚠️ No se han identificado unidades. "
                               "Comprueba la conexión con el servicio ICGC.")

                # 7. Previsualització ──────────────────────────────────────────
                st.subheader("Previsualización")
                col_map, col_leg = st.columns([3, 2])
                with col_map:
                    st.image(img_path_show, caption="Mapa geológico ICGC 1:50.000",
                             use_container_width=True)
                with col_leg:
                    if te_llegenda:
                        st.image(llegenda_path,
                                 caption="Leyenda del área (unidades identificadas)",
                                 use_container_width=True)
                    else:
                        st.info("Leyenda no disponible")

                # 8. Taula de materials ────────────────────────────────────────
                if codis_finals and (descripcions or renderer):
                    with st.expander("📋 Materiales identificados"):
                        rows = []
                        for codi in sorted(codis_finals,
                                           key=lambda c: renderer.get(c,{}).get("ordre",999)):
                            info = descripcions.get(codi, {})
                            font = "OCR" if codi in ocr_codis else "Colores"
                            rows.append({
                                "Codi":       codi,
                                "Descripción": info.get("Descripcio","—") if info else "—",
                                "Era":        info.get("Era",     "—")  if info else "—",
                                "Período":    info.get("Periode", "—")  if info else "—",
                                "Fuente":       font,
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)

                # 9. Dades vectorials ArcGIS REST (opcional) ──────────────────
                te_vector = False
                csv_path     = os.path.join(tmpdir, "icgc_unitats.csv")
                geojson_path = os.path.join(tmpdir, "icgc_recintos.geojson")
                try:
                    from shapely.geometry import shape as shapely_shape
                    pv = {
                        "f": "geojson",
                        "geometry": f"{minx},{miny},{maxx},{maxy}",
                        "geometryType": "esriGeometryEnvelope",
                        "inSR": EPSG_ICGC, "spatialRel": "esriSpatialRelIntersects",
                        "outFields": "*", "returnGeometry": "true", "outSR": EPSG_ICGC,
                    }
                    rv = requests.get(ARCGIS_BASE, params=pv, timeout=60)
                    rv.raise_for_status()
                    dv = rv.json()
                    if "error" not in dv and dv.get("features"):
                        rows_v, geoms_v = [], []
                        for feat in dv["features"]:
                            graw = feat.get("geometry")
                            props = feat.get("properties") or feat.get("attributes") or {}
                            if graw is None: continue
                            try:
                                geoms_v.append(shapely_shape(graw))
                                rows_v.append(props)
                            except Exception:
                                continue
                        if rows_v:
                            gdf = gpd.GeoDataFrame(rows_v, geometry=geoms_v, crs=f"EPSG:{EPSG_ICGC}")
                            gdf = gdf.dropna(axis=1, how="all")
                            if CAMP_CODI in gdf.columns:
                                gdf = gdf.sort_values(CAMP_CODI, kind="mergesort")
                            cols_csv = [c for c in gdf.columns if c != "geometry"]
                            gdf.drop_duplicates(
                                subset=[CAMP_CODI] if CAMP_CODI in gdf.columns else None
                            ).to_csv(csv_path, sep=";", index=False, columns=cols_csv, encoding="utf-8")
                            gdf.to_file(geojson_path, driver="GeoJSON")
                            te_vector = True
                except Exception:
                    pass

                # 10. ZIP ─────────────────────────────────────────────────────
                zip_buf = BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.write(img_path, arcname="icgc_mapa_geologia.png")
                    if fons_bytes is not None:
                        zf.write(img_path_show, arcname="icgc_mapa_compost.png")
                    if te_llegenda:
                        zf.write(llegenda_path, arcname="icgc_llegenda.png")
                    if te_vector:
                        zf.write(csv_path,      arcname="icgc_unitats.csv")
                        zf.write(geojson_path,  arcname="icgc_recintos.geojson")
                zip_buf.seek(0)
                st.download_button(
                    label="📦 Descargar todo",
                    data=zip_buf, file_name="icgc_geologia.zip", mime="application/zip",
                )

                # 11. Info tècnica ─────────────────────────────────────────────
                with st.expander("ℹ️ Información técnica"):
                    st.code(
                        f"WMS          : {WMS_BASE}\n"
                        f"Capa WMS     : {CAPA_WMS}\n"
                        f"BBOX (25831) : {minx:.1f}, {miny:.1f}, {maxx:.1f}, {maxy:.1f}\n"
                        f"Imatge       : {w_px}×{h_px} px\n"
                        f"Renderer     : {len(renderer)} entradas\n"
                        f"OCR codis    : {sorted(ocr_codis.keys())}\n"
                        f"Color codis  : {sorted(color_codis.keys())}\n"
                        f"Finals       : {sorted(codis_finals)}\n"
                        f"Vector REST  : {'sí' if te_vector else 'no disponible'}"
                    )

            except requests.exceptions.HTTPError as e:
                st.error(f"❌ Error HTTP: {e}")
            except requests.exceptions.Timeout:
                st.error("❌ Timeout. Prueba con un área más pequeña.")
            except Exception as e:
                st.error(f"❌ Error inesperado: {e}")
                import traceback
                st.code(traceback.format_exc())
