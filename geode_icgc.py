import streamlit as st
import requests
import numpy as np
import re
from difflib import get_close_matches
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


def obtenir_renderer():
    """
    Descarrega el renderer UniqueValue (colors oficials per Codi) de la capa ArcGIS.
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
        st.warning(f"No s'ha pogut descarregar el renderer ({e}). S'usaran colors neutres.")
        return {}


def obtenir_descripcions(codis):
    """Consulta ArcGIS REST per obtenir Descripcio, Era, Periode, Epoca sense geometria."""
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

def detectar_codis_ocr(img_bytes, codis_renderer):
    """
    Extreu els codis geològics de les etiquetes de text pintades al mapa usant OCR.

    Pipeline:
    1. Convertir a escala de grisos i binaritzar (text fosc < 160 → negre)
    2. Escalar x3 per millorar la detecció de text petit
    3. Tesseract PSM 11 (text dispers)
    4. Filtrar tokens per longitud i confiança
    5. Matching: exacte → case-insensitive → fuzzy (cutoff 0.72)

    Retorna: dict Codi → nombre d'ocurrències al mapa
    """
    if not TESSERACT_OK:
        return {}

    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    gray_arr = np.array(img.convert("L"))

    # Binaritzar: text fosc → negre, fons pastel → blanc
    binary = np.where(gray_arr < 160, 0, 255).astype(np.uint8)
    img_bin = Image.fromarray(binary).resize(
        (img.width * 3, img.height * 3), Image.LANCZOS
    )

    cfg = r'--psm 11 --oem 3'
    try:
        data = pytesseract.image_to_data(
            img_bin, config=cfg, output_type=pytesseract.Output.DICT
        )
    except Exception:
        return {}

    # Recollir tokens amb confiança > 25 i longitud 2-12
    raw_tokens = []
    for txt, conf in zip(data['text'], data['conf']):
        txt = txt.strip()
        if txt and conf > 25 and 2 <= len(txt) <= 12:
            raw_tokens.append(txt)

    codis_set = set(codis_renderer)
    codis_detectats = {}

    for token in raw_tokens:
        # 1. Exacte
        if token in codis_set:
            codis_detectats[token] = codis_detectats.get(token, 0) + 1
            continue
        # 2. Case-insensitive
        match_ci = next((c for c in codis_set if c.lower() == token.lower()), None)
        if match_ci:
            codis_detectats[match_ci] = codis_detectats.get(match_ci, 0) + 1
            continue
        # 3. Fuzzy
        matches = get_close_matches(token, list(codis_set), n=1, cutoff=0.72)
        if matches:
            m = matches[0]
            codis_detectats[m] = codis_detectats.get(m, 0) + 1

    return codis_detectats


# ──────────────────────────────────────────────────────────────────────────────
# DETECCIÓ DE CODIS — MÈTODE 2: COLORS (fallback i complement)
# ──────────────────────────────────────────────────────────────────────────────

def detectar_codis_colors(img_bytes, renderer, min_fraccio=0.0003, cubs_px=6):
    """
    Detecta codis a partir dels colors dominants de la imatge.

    1. Quantitza els píxels no-blancs en cubs RGB de cubs_px
    2. Selecciona clusters amb cobertura > min_fraccio
    3. Assigna cada cluster al Codi del renderer amb distància euclidiana mínima

    S'usa com a complement de l'OCR per capturar unitats grans sense etiqueta visible.
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
# FUSIÓ: OCR (principal) + COLORS (complement per a zones grans sense etiqueta)
# ──────────────────────────────────────────────────────────────────────────────

def fusionar_deteccions(ocr_codis, color_codis, total_px, min_pct_color=0.5):
    """
    Combina els resultats d'OCR i colors seguint aquesta lògica:
    - Tots els codis de l'OCR s'accepten (alta precisió).
    - Els codis NOMÉS dels colors s'accepten si cobreixen > min_pct_color% de la imatge
      (zona gran → segurament real, no soroll).
    - Els codis NOMÉS dels colors amb cobertura petita es descarten (probable fals positiu).

    Retorna: set de codis finals
    """
    finals = set(ocr_codis.keys())

    for codi, npx in color_codis.items():
        if codi in finals:
            continue  # ja cobert per OCR
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
        font_title = ImageFont.truetype(font_paths[0], 14)
        font_body  = ImageFont.truetype(font_paths[1], 13)
        font_small = ImageFont.truetype(font_paths[1], 11)
    except Exception:
        try:
            font_title = ImageFont.load_default(size=14)
            font_body  = ImageFont.load_default(size=13)
            font_small = ImageFont.load_default(size=11)
        except Exception:
            font_title = font_body = font_small = ImageFont.load_default()

    pad, sym_w, sym_h, gap = 14, 32, 18, 10
    title = "Llegenda — Unitats geològiques presents a l'àrea"

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
        row_heights.append(max(sym_h + 4, n * 15 + 6))

    img_w = pad * 2 + sym_w + gap + max_w + pad
    img_h = pad + 20 + pad + sum(rh + 6 for rh in row_heights) + pad

    img  = Image.new("RGB", (int(img_w), int(img_h)), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((pad, pad), title, fill=(20, 20, 20), font=font_title)

    y = pad + 20 + pad
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
        ty += 16
        if desc not in ("","None","nan"):
            draw.text((tx, ty), desc, fill=(50, 50, 50), font=font_small)
            ty += 14
        age = era if era not in ("","None","nan") else ""
        if per not in ("","None","nan") and age:
            age += f", {per}"
        elif per not in ("","None","nan"):
            age = per
        if age:
            draw.text((tx, ty), age, fill=(110, 110, 110), font=font_small)

        y += rh + 6

    img.save(output_path)
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# INTERFÍCIE STREAMLIT
# ──────────────────────────────────────────────────────────────────────────────

st.title("🗺️ Generador de Cartografia Geològica (ICGC)")

metode_info = "OCR (Tesseract) + colors dominants" if TESSERACT_OK else "colors dominants (Tesseract no instal·lat)"
st.markdown(f"""
Consulta el **WMS Geologia Territorial 1:50.000 de l'ICGC** i genera:

1. **Mapa geològic** PNG de l'àrea seleccionada.
2. **Llegenda específica** construïda identificant les unitats presents:
   - **Mètode principal**: OCR sobre les etiquetes de text del mapa *(alta precisió)*.
   - **Complement**: anàlisi de colors dominants *(captura zones grans sense etiqueta)*.
   - Mètode actiu: `{metode_info}`
3. Arxius **CSV** i **GeoJSON** si el servei vectorial és accessible.

> Coordenades **ETRS89 UTM zona 31N (EPSG:25831)**
""")

if not TESSERACT_OK:
    st.warning("⚠️ `pytesseract` no instal·lat. S'usarà només l'anàlisi de colors. "
               "Per activar OCR: `pip install pytesseract` + `apt install tesseract-ocr`")

with st.sidebar:
    st.header("Paràmetres d'entrada")
    st.info("Coordenades ETRS89 UTM zona 31N (EPSG:25831)")
    st.subheader("Centre del rectangle (UTM)")
    c1, c2 = st.columns(2)
    x_centre = c1.number_input("X (m)", value=431_200.0, format="%.1f")
    y_centre = c2.number_input("Y (m)", value=4_582_200.0, format="%.1f")
    st.subheader("Dimensions del rectangle")
    d1, d2 = st.columns(2)
    ample = d1.number_input("Ample (m)", value=5_000.0, min_value=100.0)
    alt   = d2.number_input("Alt (m)",   value=4_000.0, min_value=100.0)
    executar = st.button("🚀 Generar mapa i llegenda", type="primary")

if executar:
    with st.spinner("Connectant amb ICGC…"):
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                minx, miny, maxx, maxy = bbox_utm_25831(x_centre, y_centre, ample, alt)
                w_px, h_px = calcular_px(ample, alt)
                total_px = w_px * h_px

                # 1. Imatge WMS ────────────────────────────────────────────────
                with st.spinner("Descarregant mapa WMS…"):
                    img_bytes = obtenir_imatge_wms(minx, miny, maxx, maxy, w_px, h_px)
                img_path = os.path.join(tmpdir, "icgc_mapa.png")
                with open(img_path, "wb") as f:
                    f.write(img_bytes)

                # 2. Renderer de colors ────────────────────────────────────────
                with st.spinner("Descarregant renderer de colors…"):
                    renderer = obtenir_renderer()

                codis_renderer = set(renderer.keys())

                # 3a. OCR — mètode principal ───────────────────────────────────
                ocr_codis = {}
                if TESSERACT_OK:
                    with st.spinner("Aplicant OCR sobre les etiquetes del mapa…"):
                        ocr_codis = detectar_codis_ocr(img_bytes, codis_renderer)

                # 3b. Colors — mètode complement ──────────────────────────────
                with st.spinner("Analitzant colors dominants de la imatge…"):
                    color_codis = detectar_codis_colors(img_bytes, renderer)

                # 3c. Fusió ───────────────────────────────────────────────────
                codis_finals = fusionar_deteccions(ocr_codis, color_codis, total_px)

                # 4. Descripcions per als codis detectats ─────────────────────
                descripcions = {}
                if codis_finals:
                    with st.spinner(f"Obtenint descripcions de {len(codis_finals)} unitats…"):
                        descripcions = obtenir_descripcions(list(codis_finals))

                # 5. Llegenda ─────────────────────────────────────────────────
                llegenda_path = os.path.join(tmpdir, "icgc_llegenda.png")
                te_llegenda = False
                if codis_finals:
                    r = construir_llegenda(codis_finals, renderer, descripcions, llegenda_path)
                    te_llegenda = (r is not None)

                # 6. Missatge d'èxit ───────────────────────────────────────────
                if codis_finals:
                    ocr_n   = len(ocr_codis)
                    color_n = len([c for c in codis_finals if c not in ocr_codis])
                    st.success(
                        f"✅ **{len(codis_finals)} unitats identificades** — "
                        f"{ocr_n} via OCR · {color_n} via colors dominants"
                    )
                else:
                    st.warning("⚠️ No s'han identificat unitats. Comprova la connexió amb el servei ICGC.")

                # 7. Previsualització ──────────────────────────────────────────
                st.subheader("Previsualització")
                col_map, col_leg = st.columns([2, 1])
                with col_map:
                    st.image(img_path, caption="Mapa geològic ICGC 1:50.000",
                             use_container_width=True)
                with col_leg:
                    if te_llegenda:
                        st.image(llegenda_path,
                                 caption="Llegenda de l'àrea (unitats identificades)",
                                 use_container_width=True)
                    else:
                        st.info("Llegenda no disponible")

                # 8. Taula de materials ────────────────────────────────────────
                if codis_finals and (descripcions or renderer):
                    with st.expander("📋 Materials identificats"):
                        rows = []
                        for codi in sorted(codis_finals,
                                           key=lambda c: renderer.get(c,{}).get("ordre",999)):
                            info = descripcions.get(codi, {})
                            font = "OCR" if codi in ocr_codis else "Colors"
                            rows.append({
                                "Codi":       codi,
                                "Descripció": info.get("Descripcio","—") if info else "—",
                                "Era":        info.get("Era",     "—")  if info else "—",
                                "Període":    info.get("Periode", "—")  if info else "—",
                                "Font":       font,
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
                    zf.write(img_path, arcname="icgc_mapa.png")
                    if te_llegenda:
                        zf.write(llegenda_path, arcname="icgc_llegenda.png")
                    if te_vector:
                        zf.write(csv_path,      arcname="icgc_unitats.csv")
                        zf.write(geojson_path,  arcname="icgc_recintos.geojson")
                zip_buf.seek(0)
                st.download_button(
                    label="📦 Descarregar tot",
                    data=zip_buf, file_name="icgc_geologia.zip", mime="application/zip",
                )

                # 11. Info tècnica ─────────────────────────────────────────────
                with st.expander("ℹ️ Informació tècnica"):
                    st.code(
                        f"WMS          : {WMS_BASE}\n"
                        f"Capa WMS     : {CAPA_WMS}\n"
                        f"BBOX (25831) : {minx:.1f}, {miny:.1f}, {maxx:.1f}, {maxy:.1f}\n"
                        f"Imatge       : {w_px}×{h_px} px\n"
                        f"Renderer     : {len(renderer)} entrades\n"
                        f"OCR codis    : {sorted(ocr_codis.keys())}\n"
                        f"Color codis  : {sorted(color_codis.keys())}\n"
                        f"Finals       : {sorted(codis_finals)}\n"
                        f"Vector REST  : {'sí' if te_vector else 'no disponible'}"
                    )

            except requests.exceptions.HTTPError as e:
                st.error(f"❌ Error HTTP: {e}")
            except requests.exceptions.Timeout:
                st.error("❌ Timeout. Prova amb una àrea més petita.")
            except Exception as e:
                st.error(f"❌ Error inesperat: {e}")
                import traceback
                st.code(traceback.format_exc())
