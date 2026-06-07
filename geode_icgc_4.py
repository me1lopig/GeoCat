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

st.set_page_config(page_title="Visor Geología ICGC", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
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
# FUNCIONES DE SERVICIO
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
        raise RuntimeError(f"WMS no ha retornado imagen. CT:{ct}\n{resp.text[:300]}")
    return resp.content


def obtenir_imatge_fons(minx, miny, maxx, maxy, ample_px, alt_px, servei_key):
    servei = FONS_SERVEIS.get(servei_key)
    if servei is None:
        return None
    params = {
        "SERVICE": "WMS", "VERSION": servei["ver"], "REQUEST": "GetMap",
        "LAYERS": servei["layer"], "STYLES": "", "FORMAT": servei["fmt"],
        "TRANSPARENT": "false",
        servei["srs_key"]: f"EPSG:{EPSG_ICGC}",
        "BBOX": f"{minx},{miny},{maxx},{maxy}",
        "WIDTH": int(ample_px), "HEIGHT": int(alt_px),
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
    fons = Image.open(BytesIO(fons_bytes)).convert("RGBA")
    geol = Image.open(BytesIO(geol_bytes)).convert("RGBA")
    if fons.size != geol.size:
        geol = geol.resize(fons.size, Image.LANCZOS)
    r, g, b, a = geol.split()
    arr_r = np.array(r); arr_g = np.array(g); arr_b = np.array(b)
    # Píxeles blancos (fondo neutro del WMS geología) → transparentes
    blanc_mask = (arr_r > 248) & (arr_g > 248) & (arr_b > 248)
    arr_a = np.array(a).astype(np.float32)
    arr_a[blanc_mask] = 0
    arr_a[~blanc_mask] = arr_a[~blanc_mask] * opacitat
    a_nou = Image.fromarray(arr_a.astype(np.uint8))
    geol_mod = Image.merge("RGBA", (r, g, b, a_nou))
    resultat = fons.copy()
    resultat.paste(geol_mod, (0, 0), geol_mod)
    buf = BytesIO()
    resultat.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def obtenir_unitats_bbox(minx, miny, maxx, maxy):
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
# DETECCIÓN OCR — MÉTODO FALLBACK
# ──────────────────────────────────────────────────────────────────────────────

def _match_token(token, codis_set, cutoff=0.72):
    if not token:
        return None
    if token in codis_set:
        return token
    m = next((c for c in codis_set if c.lower() == token.lower()), None)
    if m:
        return m
    subs = {'O': 'Q', 'Q': 'O', 'l': '1', '0': 'O', '|': 'l'}
    if token[0] in subs:
        tc = subs[token[0]] + token[1:]
        if tc in codis_set:
            return tc
        m = next((c for c in codis_set if c.lower() == tc.lower()), None)
        if m:
            return m
    candidates = get_close_matches(token, list(codis_set), n=3, cutoff=cutoff)
    if candidates:
        best, best_sc = None, 0.0
        for cand in candidates:
            sc = SequenceMatcher(None, token, cand).ratio()
            if token[0].lower() == cand[0].lower():
                sc *= 1.1
            if sc > best_sc:
                best, best_sc = cand, sc
        if best and best_sc >= cutoff:
            return best
    return None


def detectar_codis_ocr(img_bytes, codis_renderer):
    if not TESSERACT_OK:
        return {}
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    gray_arr = np.array(img.convert("L"))
    cfg = r'--psm 11 --oem 3'
    codis_set = set(codis_renderer)
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
    return {
        codi: info["max_conf"]
        for codi, info in codi_info.items()
        if len(info["umbrals"]) >= 2 or info["max_conf"] > 60
    }


def detectar_codis_colors(img_bytes, renderer, min_fraccio=0.0003, cubs_px=6):
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


def fusionar_deteccions(ocr_codis, color_codis, total_px, min_pct_color=0.5):
    finals = set(ocr_codis.keys())
    for codi, npx in color_codis.items():
        if codi in finals:
            continue  # ya cubierto por OCR
        pct = npx / total_px * 100
        if pct >= min_pct_color:
            finals.add(codi)
    return finals


# ──────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE LA LEYENDA
# ──────────────────────────────────────────────────────────────────────────────

def construir_llegenda(codis_finals, renderer, descripcions, output_path):
    if not codis_finals:
        return None
    codis_ordenats = sorted(
        codis_finals,
        key=lambda c: renderer.get(c, {}).get("ordre", 999)
    )
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

    max_w = tw(title, font_title)
    row_heights = []
    for codi in codis_ordenats:
        info = descripcions.get(codi, {})
        desc = str(info.get("Descripcio", "")) if info else ""
        era  = str(info.get("Era", ""))         if info else ""
        for txt, font in [(codi, font_body), (desc, font_small), (era, font_small)]:
            if txt and txt not in ("None", "nan"):
                max_w = max(max_w, tw(txt, font))
        n = 1
        if desc not in ("", "None", "nan"): n += 1
        if era  not in ("", "None", "nan"): n += 1
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
        desc = str(info.get("Descripcio", "")) if info else ""
        era  = str(info.get("Era", ""))         if info else ""
        per  = str(info.get("Periode", ""))     if info else ""
        color = renderer.get(codi, {}).get("color", (190, 190, 190))
        draw.rectangle(
            [pad, int(y + rh/2 - sym_h/2), pad + sym_w, int(y + rh/2 + sym_h/2)],
            fill=color, outline=(80, 80, 80)
        )
        tx, ty = pad + sym_w + gap, int(y)
        draw.text((tx, ty), codi, fill=(20, 20, 20), font=font_body);  ty += 20
        if desc not in ("", "None", "nan"):
            draw.text((tx, ty), desc, fill=(50, 50, 50), font=font_small); ty += 17
        age = era if era not in ("", "None", "nan") else ""
        if per not in ("", "None", "nan") and age:
            age += f", {per}"
        elif per not in ("", "None", "nan"):
            age = per
        if age:
            draw.text((tx, ty), age, fill=(110, 110, 110), font=font_small)
        y += rh + 8

    img.save(output_path)
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN CENTRAL DE VISUALIZACIÓN (usa session_state)
# ──────────────────────────────────────────────────────────────────────────────

def mostrar_resultados(opacitat, fons_key, fons_bytes_nuevo=None):
    ss = st.session_state
    img_bytes    = ss["geol_bytes"]
    renderer     = ss["renderer"]
    codis_finals = ss["codis_finals"]
    descripcions = ss["descripcions"]
    unitats_api  = ss["unitats_api"]
    metode_used  = ss["metode_used"]
    w_px         = ss["w_px"]
    h_px         = ss["h_px"]

    # Actualizar fondo si hay uno nuevo
    if fons_bytes_nuevo is not None:
        ss["fons_bytes"] = fons_bytes_nuevo
        ss["fons_key"]   = fons_key
    elif fons_key != ss.get("fons_key"):
        # El fondo cambió pero no se descargó — limpiar el anterior
        ss["fons_bytes"] = None
        ss["fons_key"]   = fons_key

    fons_bytes = ss.get("fons_bytes")
    ss["opacitat"] = opacitat

    with tempfile.TemporaryDirectory() as tmpdir:
        # Guardar imagen geología
        img_path = os.path.join(tmpdir, "icgc_mapa.png")
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        # Componer geología + fondo
        if fons_bytes is not None:
            img_comp = compondre_imatges(fons_bytes, img_bytes, opacitat / 100)
            img_path_show = os.path.join(tmpdir, "icgc_compost.png")
            with open(img_path_show, "wb") as f:
                f.write(img_comp)
        else:
            img_path_show = img_path

        # Leyenda
        llegenda_path = os.path.join(tmpdir, "icgc_llegenda.png")
        te_llegenda = False
        if codis_finals:
            r = construir_llegenda(codis_finals, renderer, descripcions, llegenda_path)
            te_llegenda = (r is not None)

        # Mensaje de estado
        if codis_finals:
            st.success(
                f"✅ **{len(codis_finals)} unidades geológicas identificadas** — "
                f"Método: *{metode_used}*"
            )

        # Previsualización
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

        # Tabla materiales
        if codis_finals and descripcions:
            with st.expander("📋 Materiales identificados"):
                rows = []
                for codi in sorted(
                    codis_finals,
                    key=lambda c: (
                        descripcions.get(c, {}).get("Ordre", 999)
                        if unitats_api else
                        renderer.get(c, {}).get("ordre", 999)
                    )
                ):
                    info = descripcions.get(codi, {})
                    rows.append({
                        "Codi":        codi,
                        "Descripción": info.get("Descripcio", "—") if info else "—",
                        "Era":         info.get("Era",      "—")   if info else "—",
                        "Período":     info.get("Periode",  "—")   if info else "—",
                        "Época":       info.get("Epoca",    "—")   if info else "—",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # ZIP descarga
        te_vector    = ss.get("te_vector", False)
        csv_path     = os.path.join(tmpdir, "icgc_unitats.csv")
        geojson_path = os.path.join(tmpdir, "icgc_recintos.geojson")
        if te_vector:
            with open(csv_path,     "wb") as f: f.write(ss["csv_bytes"])
            with open(geojson_path, "wb") as f: f.write(ss["geojson_bytes"])

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

        # Info técnica
        with st.expander("ℹ️ Información técnica"):
            st.code(
                f"WMS          : {WMS_BASE}\n"
                f"Capa WMS     : {CAPA_WMS}\n"
                f"BBOX (25831) : {ss['minx']:.1f}, {ss['miny']:.1f}, "
                f"{ss['maxx']:.1f}, {ss['maxy']:.1f}\n"
                f"Imagen       : {w_px}×{h_px} px\n"
                f"Método ID    : {metode_used}\n"
                f"Unidades     : {len(codis_finals)}\n"
                f"Codis        : {sorted(codis_finals)}\n"
                f"Fondo activo : {ss.get('fons_key', '—')}\n"
                f"Opacidad     : {opacitat}%\n"
                f"Vector REST  : {'sí' if te_vector else 'no disponible'}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# INTERFAZ STREAMLIT
# ──────────────────────────────────────────────────────────────────────────────

metode_info = "OCR (Tesseract) + colores dominantes" if TESSERACT_OK else "colores dominantes (Tesseract no instalado)"
st.title("🗺️ Generador de Cartografía Geológica (ICGC)")
st.markdown(f"""
Consulta el **WMS Geología Territorial 1:50.000 del ICGC** y genera:

1. **Mapa geológico** PNG del área seleccionada.
2. **Leyenda específica** construida identificando las unidades presentes:
   - **Método principal**: consulta directa API ArcGIS REST *(precisión 100%)*.
   - **Fallback**: OCR + análisis de colores dominantes si la API no responde.
   - Método activo: `{metode_info}`
3. Diseñado según directriz ITQ404.

> Coordenadas **ETRS89 UTM zona 31N (EPSG:25831)**
""")

if not TESSERACT_OK:
    st.warning("⚠️ `pytesseract` no instalado. Se usará solo el análisis de colores. "
               "Para activar OCR: `pip install pytesseract` + `apt install tesseract-ocr`")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parámetros de entrada")
    st.info("Coordenadas ETRS89 UTM zona 31N (EPSG:25831)")
    st.subheader("Centro del rectángulo (UTM)")
    c1, c2 = st.columns(2)
    x_centre = c1.number_input("X (m)", value=431_200.0, format="%.1f")
    y_centre = c2.number_input("Y (m)", value=4_582_200.0, format="%.1f")
    st.subheader("Dimensiones del rectángulo")
    d1, d2 = st.columns(2)
    ample = d1.number_input("Ample (m)", value=5_000.0, min_value=100.0)
    alt   = d2.number_input("Alt (m)",   value=4_000.0, min_value=100.0)

    st.divider()
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

    st.divider()
    executar        = st.button("🚀 Generar mapa y leyenda", type="primary")
    hay_geologia    = "geol_bytes" in st.session_state
    actualizar_fons = st.button(
        "🔄 Actualizar fondo / opacidad",
        disabled=not hay_geologia,
        help="Cambia el fondo sin volver a descargar la geología" if hay_geologia
             else "Primero genera el mapa con el botón superior"
    )

    # Indicador de estado del caché
    if hay_geologia:
        ss = st.session_state
        st.caption(
            f"✅ Geología en memoria: {ss.get('w_px',0)}×{ss.get('h_px',0)} px  \n"
            f"Área: ({ss.get('minx',0):.0f}, {ss.get('miny',0):.0f}) — "
            f"({ss.get('maxx',0):.0f}, {ss.get('maxy',0):.0f})"
        )

# ── Acción: Generar mapa completo ─────────────────────────────────────────────
if executar:
    with st.spinner("Conectando con ICGC…"):
        try:
            minx, miny, maxx, maxy = bbox_utm_25831(x_centre, y_centre, ample, alt)
            w_px, h_px = calcular_px(ample, alt)
            total_px   = w_px * h_px

            # 1. Imagen WMS geología
            with st.spinner("Descargando mapa WMS…"):
                img_bytes = obtenir_imatge_wms(minx, miny, maxx, maxy, w_px, h_px)

            # 2. Renderer de colores
            with st.spinner("Descargando renderer de colores…"):
                renderer = obtenir_renderer()
            codis_renderer = set(renderer.keys())

            # 3. Identificación de unidades — método prioritario: API directa
            unitats_api  = None
            ocr_codis    = {}
            color_codis  = {}
            metode_used  = ""

            with st.spinner("Consultando unidades geológicas del área (API)…"):
                unitats_api = obtenir_unitats_bbox(minx, miny, maxx, maxy)

            if unitats_api is not None:
                # MÉTODO PRINCIPAL: API directa
                metode_used  = "API directa (ArcGIS REST)"
                codis_finals = {u["Codi"] for u in unitats_api if u.get("Codi")}
                descripcions = {u["Codi"]: u for u in unitats_api if u.get("Codi")}
            else:
                # FALLBACK: OCR + colores
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

            # 4. Fondo
            fons_bytes = None
            if fons_key != "Sin fondo (solo geología)":
                with st.spinner(f"Descargando capa de fondo: {fons_key}…"):
                    fons_bytes = obtenir_imatge_fons(
                        minx, miny, maxx, maxy, w_px, h_px, fons_key
                    )
                if fons_bytes is None:
                    st.warning(f"⚠️ No se ha podido descargar '{fons_key}'. "
                               "Se mostrará solo la geología.")

            # 5. Datos vectoriales (opcional)
            te_vector    = False
            csv_bytes    = b""
            geojson_bytes = b""
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
                        graw  = feat.get("geometry")
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
                        with tempfile.TemporaryDirectory() as td:
                            csv_p = os.path.join(td, "u.csv")
                            geo_p = os.path.join(td, "u.geojson")
                            cols_csv = [c for c in gdf.columns if c != "geometry"]
                            gdf.drop_duplicates(
                                subset=[CAMP_CODI] if CAMP_CODI in gdf.columns else None
                            ).to_csv(csv_p, sep=";", index=False, columns=cols_csv, encoding="utf-8")
                            gdf.to_file(geo_p, driver="GeoJSON")
                            with open(csv_p, "rb") as f: csv_bytes = f.read()
                            with open(geo_p, "rb") as f: geojson_bytes = f.read()
                        te_vector = True
            except Exception:
                pass

            # 6. Guardar todo en session_state
            st.session_state.update({
                "geol_bytes":    img_bytes,
                "renderer":      renderer,
                "codis_finals":  codis_finals,
                "descripcions":  descripcions,
                "unitats_api":   unitats_api,
                "metode_used":   metode_used,
                "w_px": w_px, "h_px": h_px,
                "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy,
                "fons_bytes":    fons_bytes,
                "fons_key":      fons_key,
                "opacitat":      opacitat,
                "te_vector":     te_vector,
                "csv_bytes":     csv_bytes,
                "geojson_bytes": geojson_bytes,
            })

            # 7. Mostrar resultados
            mostrar_resultados(opacitat, fons_key, fons_bytes)

        except requests.exceptions.HTTPError as e:
            st.error(f"❌ Error HTTP: {e}")
        except requests.exceptions.Timeout:
            st.error("❌ Timeout. Prueba con un área más pequeña.")
        except Exception as e:
            st.error(f"❌ Error inesperado: {e}")
            import traceback
            st.code(traceback.format_exc())


# ── Acción: Actualizar solo fondo / opacidad ──────────────────────────────────
elif actualizar_fons and "geol_bytes" in st.session_state:
    ss = st.session_state
    fons_bytes_nuevo = None

    # ¿Ha cambiado el fondo?
    if fons_key != ss.get("fons_key"):
        if fons_key != "Sin fondo (solo geología)":
            with st.spinner(f"Descargando nueva capa de fondo: {fons_key}…"):
                fons_bytes_nuevo = obtenir_imatge_fons(
                    ss["minx"], ss["miny"], ss["maxx"], ss["maxy"],
                    ss["w_px"], ss["h_px"], fons_key
                )
            if fons_bytes_nuevo is None:
                st.warning(f"⚠️ No se ha podido descargar '{fons_key}'. "
                           "Se mostrará la geología sin fondo.")
        else:
            # "Sin fondo" seleccionado → limpiar fondo anterior
            ss["fons_bytes"] = None
            ss["fons_key"]   = fons_key

    # Mostrar con el nuevo fondo (o solo nueva opacidad)
    mostrar_resultados(opacitat, fons_key, fons_bytes_nuevo)


# ── Si hay datos en session_state pero no se ha pulsado ningún botón ──────────
elif "geol_bytes" in st.session_state and not executar and not actualizar_fons:
    mostrar_resultados(
        st.session_state.get("opacitat", opacitat),
        st.session_state.get("fons_key", fons_key),
    )
