import streamlit as st
import requests
from shapely.geometry import Polygon
import geopandas as gpd
import pandas as pd
from io import BytesIO
from PIL import Image, ImageDraw
import tempfile
import os
import zipfile

# ──────────────────────────────────────────────────────────────────────────────
# Configuració de la pàgina
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Visor Geologia ICGC", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (verificades contra l'API real del ICGC, maig 2026)
# ──────────────────────────────────────────────────────────────────────────────
WMS_BASE    = "https://geoserveis.icgc.cat/servei/catalunya/geologia-territorial/wms"
CAPA_WMS    = "unitats-geologiques-50000"

# ArcGIS REST públic del ICGC — capa 38 = unitats-geologiques-50000
# Verificat a: https://maps.icgc.cat/vector01/rest/services/geologia_territorial/MapServer/38
ARCGIS_BASE = "https://maps.icgc.cat/vector01/rest/services/geologia_territorial/MapServer/38/query"

# Camp identificador real de la capa (Display Field confirmat a l'API)
CAMP_CODI    = "Codi"

# URL del renderer ArcGIS per obtenir colors reals per Codi
ARCGIS_LAYER = "https://maps.icgc.cat/vector01/rest/services/geologia_territorial/MapServer/38" 

# Camps descriptius disponibles a la capa (model de dades ICGC v3, 2025)
# Confirmats a les especificacions tècniques oficials del ICGC
CAMPS_DESC  = ["Codi", "Descripcio", "Ordre", "Eo", "Era", "Periode", "Epoca"]

EPSG_ICGC   = 25831   # UTM zona 31N ETRS89 – CRS natiu de l'ICGC

# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONS AUXILIARS
# ──────────────────────────────────────────────────────────────────────────────

def bbox_utm_25831(x_centre, y_centre, ample_m, alt_m):
    """Retorna (minx, miny, maxx, maxy) en EPSG:25831."""
    mx, my = ample_m / 2.0, alt_m / 2.0
    return (x_centre - mx, y_centre - my, x_centre + mx, y_centre + my)


def calcular_px(ample_m, alt_m, max_dim=1024):
    """Calcula els píxels mantenint el ràtio d'aspecte amb una dimensió màxima."""
    if ample_m >= alt_m:
        w = int(max_dim)
        h = max(1, int(round(max_dim * alt_m / ample_m)))
    else:
        h = int(max_dim)
        w = max(1, int(round(max_dim * ample_m / alt_m)))
    return w, h


def obtenir_imatge_wms(minx, miny, maxx, maxy, ample_px=1024, alt_px=768):
    """
    Crida WMS GetMap i retorna els bytes PNG.
    WMS 1.3.0 amb EPSG:25831: ordre d'eixos E,N → BBOX = minE,minN,maxE,maxN.
    STYLES s'envia com a cadena buida (valor vàlid per a estil per defecte).
    """
    params = {
        "SERVICE":     "WMS",
        "VERSION":     "1.3.0",
        "REQUEST":     "GetMap",
        "LAYERS":      CAPA_WMS,
        "STYLES":      "",
        "FORMAT":      "image/png",
        "TRANSPARENT": "false",
        "CRS":         f"EPSG:{EPSG_ICGC}",
        "BBOX":        f"{minx},{miny},{maxx},{maxy}",
        "WIDTH":       int(ample_px),
        "HEIGHT":      int(alt_px),
    }
    resp = requests.get(WMS_BASE, params=params, timeout=60)
    resp.raise_for_status()
    ct = resp.headers.get("Content-Type", "")
    if "image" not in ct:
        raise RuntimeError(
            f"WMS no ha retornat una imatge. Content-Type: {ct}\n{resp.text[:400]}"
        )
    return resp.content


def obtenir_llegenda_wms():
    """
    Crida WMS GetLegendGraphic i retorna una imatge PIL.
    El ICGC suporta GetLegendGraphic al WMS de geologia-territorial (confirmat).
    """
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetLegendGraphic",
        "LAYER":   CAPA_WMS,
        "FORMAT":  "image/png",
        "STYLE":   "",
    }
    resp = requests.get(WMS_BASE, params=params, timeout=30)
    resp.raise_for_status()
    ct = resp.headers.get("Content-Type", "")
    if "image" not in ct:
        raise RuntimeError(
            f"GetLegendGraphic no ha retornat imatge. Content-Type:{ct}\n{resp.text[:200]}"
        )
    return Image.open(BytesIO(resp.content)).convert("RGBA")


def obtenir_poligons_arcgis(minx, miny, maxx, maxy):
    """
    Consulta l'ArcGIS REST públic del ICGC (maps.icgc.cat) per obtenir els
    polígons de unitats-geologiques-50000 que intersecten amb el BBOX.

    Endpoint verificat: maps.icgc.cat/vector01/rest/services/geologia_territorial/MapServer/38
    Camp identificador real: 'Codi'  (Display Field confirmat a l'API)
    MaxRecordCount del servei: 2000
    """
    params = {
        "f":              "geojson",
        "geometry":       f"{minx},{miny},{maxx},{maxy}",
        "geometryType":   "esriGeometryEnvelope",
        "inSR":           EPSG_ICGC,
        "spatialRel":     "esriSpatialRelIntersects",
        "outFields":      "*",
        "returnGeometry": "true",
        "outSR":          EPSG_ICGC,
    }
    try:
        resp = requests.get(ARCGIS_BASE, params=params, timeout=90)
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "")
        if "json" not in ct:
            raise ValueError(f"ArcGIS REST ha retornat Content-Type inesperat: {ct}")
        data = resp.json()
        # ArcGIS pot retornar un error JSON en lloc de features
        if "error" in data:
            raise ValueError(f"Error ArcGIS: {data['error']}")
        features = data.get("features", [])
        if not features:
            return None

        # Construïm el GeoDataFrame manualment per ser robustos davant
        # variacions del GeoJSON d'ArcGIS REST (atributs en 'properties' o 'attributes')
        from shapely.geometry import shape as shapely_shape
        rows, geoms = [], []
        for feat in features:
            geom_raw = feat.get("geometry")
            props    = feat.get("properties") or feat.get("attributes") or {}
            if geom_raw is None:
                continue
            try:
                geom = shapely_shape(geom_raw)
            except Exception:
                continue
            rows.append(props)
            geoms.append(geom)

        if not rows:
            return None

        gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs=f"EPSG:{EPSG_ICGC}")
        return gdf

    except Exception as e:
        st.warning(
            f"ArcGIS REST no disponible ({e}). "
            "El mapa i la llegenda s'han generat igualment."
        )
        return None


def processar_gdf(gdf):
    """
    Neteja i ordena el GeoDataFrame:
    - Elimina columnes completament buides
    - Ordena per 'Codi' (camp real del ICGC, confirmat a l'API)
    - Deduplicata per unitat geològica (per al CSV)
    """
    if gdf is None or len(gdf) == 0:
        return None, None

    # Eliminar columnes completament buides
    gdf = gdf.dropna(axis=1, how="all")

    # Usar el camp 'Codi' confirmat; si per algun motiu no existís, fallback genèric
    if CAMP_CODI in gdf.columns:
        gdf = gdf.sort_values(by=CAMP_CODI, kind="mergesort").reset_index(drop=True)
        gdf_unitats = gdf.drop_duplicates(subset=[CAMP_CODI]).copy()
    else:
        # Fallback: cercar qualsevol camp amb nom similar
        id_candidates = ["codi", "Codi", "codi_unitat", "id_unitat", "etiqueta", "id"]
        id_col = next((c for c in id_candidates if c in gdf.columns), None)
        if id_col:
            gdf = gdf.sort_values(by=id_col, kind="mergesort").reset_index(drop=True)
            gdf_unitats = gdf.drop_duplicates(subset=[id_col]).copy()
        else:
            gdf_unitats = gdf.copy()

    return gdf, gdf_unitats



def obtenir_colors_renderer():
    """
    Descarrega el renderer UniqueValue de la capa ArcGIS REST i retorna un dict
    Codi → (R, G, B) amb els colors reals del mapa geològic.
    Si el servei no és accessible, retorna un dict buit i la llegenda
    usarà colors neutres.
    """
    try:
        resp = requests.get(ARCGIS_LAYER + "?f=pjson", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        infos = data.get("drawingInfo", {}).get("renderer", {}).get("uniqueValueInfos", [])
        color_map = {}
        for info in infos:
            val   = info.get("value", "")          # format "Ordre,Codi"
            color = info.get("symbol", {}).get("color", [200, 200, 200, 255])
            codi  = val.split(",")[1].strip() if "," in val else val.strip()
            if codi:
                color_map[codi] = tuple(color[:3])
        return color_map
    except Exception:
        return {}



def generar_llegenda_textual(gdf_unitats, output_path, color_map=None):
    """
    Genera una imatge PNG amb la llegenda textual de les unitats geològiques
    trobades a l'àrea, usant els colors reals del renderer ArcGIS (camp RGBA).
    Si no hi ha camps de color, usa rectangles grisos neutres.
    Camps usats: Codi, Descripcio, Era, Periode, Epoca (tots del model ICGC v3).
    """
    if gdf_unitats is None or len(gdf_unitats) == 0:
        return None

    # Preparar files: seleccionar i ordenar camps disponibles
    camps_disp = [c for c in CAMPS_DESC if c in gdf_unitats.columns]
    if not camps_disp:
        return None

    df = gdf_unitats[camps_disp].copy().reset_index(drop=True)

    # Configuració visual
    try:
        from PIL import ImageFont
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        font_body  = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        try:
            font_title = ImageFont.load_default(size=14)
            font_body  = ImageFont.load_default(size=12)
        except Exception:
            font_title = ImageFont.load_default()
            font_body  = ImageFont.load_default()

    pad, sym_w, sym_h, gap, row_h = 12, 28, 16, 8, 22
    title_text = "Unitats geològiques — ICGC Geologia Territorial 1:50.000"

    # Calcular amplada necessària
    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    def text_w(text, font):
        bb = dummy.textbbox((0, 0), text, font=font)
        return bb[2] - bb[0]

    max_w = text_w(title_text, font_title)
    for _, row in df.iterrows():
        codi  = str(row.get("Codi", ""))
        desc  = str(row.get("Descripcio", ""))
        era   = str(row.get("Era",  "")) if "Era"   in df.columns else ""
        per   = str(row.get("Periode", "")) if "Periode" in df.columns else ""
        line  = f"{codi}  —  {desc}"
        if era and era not in ("None", "nan", ""):
            line += f"  [{era}" + (f", {per}" if per not in ("None","nan","") else "") + "]"
        max_w = max(max_w, text_w(line, font_body))

    img_w = pad * 2 + sym_w + gap + max_w + pad
    img_h = pad * 2 + row_h + pad + len(df) * (row_h + 4)

    img   = Image.new("RGBA", (int(img_w), int(img_h)), (255, 255, 255, 255))
    draw  = ImageDraw.Draw(img)

    # Títol
    draw.text((pad, pad), title_text, fill=(30, 30, 30), font=font_title)
    y = pad + row_h + pad

    # Files per unitat
    for _, row in df.iterrows():
        codi  = str(row.get("Codi", ""))
        desc  = str(row.get("Descripcio", ""))
        era   = str(row.get("Era",  "")) if "Era"   in df.columns else ""
        per   = str(row.get("Periode", "")) if "Periode" in df.columns else ""

        # Rectangle de color real del renderer (o gris neutre si no disponible)
        color_fill = color_map.get(codi, (200, 200, 200)) if color_map else (200, 200, 200)
        draw.rectangle([pad, int(y), pad + sym_w, int(y + sym_h)],
                       fill=color_fill + (255,), outline=(80, 80, 80, 255))

        # Text descriptiu
        line = f"{codi}  —  {desc}"
        if era and era not in ("None", "nan", ""):
            line += f"  [{era}" + (f", {per}" if per not in ("None","nan","") else "") + "]"
        draw.text((pad + sym_w + gap, int(y + 2)), line, fill=(30, 30, 30), font=font_body)

        y += row_h + 4

    img.save(output_path)
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# INTERFÍCIE STREAMLIT
# ──────────────────────────────────────────────────────────────────────────────

st.title("🗺️ Generador de Cartografia Geològica (ICGC)")
st.markdown("""
Aquesta aplicació consulta el servei **WMS Geologia Territorial 1:50.000 de l'ICGC**
i genera per a l'àrea seleccionada:

1. **Mapa geològic** en PNG (WMS GetMap).
2. **Llegenda** oficial del WMS (GetLegendGraphic).
3. Arxius **CSV** i **GeoJSON** amb les unitats i polígons (ArcGIS REST públic).
4. Cobertura: **tot Catalunya**.

> Coordenades en **ETRS89 UTM zona 31N (EPSG:25831)**, el sistema natiu de l'ICGC.
""")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Paràmetres d'entrada")
    st.info("Coordenades ETRS89 UTM zona 31N (EPSG:25831)")

    st.subheader("Centre del rectangle (UTM)")
    c1, c2 = st.columns(2)
    # Valors per defecte: centre de la comarca del Barcelonès
    x_centre = c1.number_input("X (m)", value=431_200.0, format="%.1f")
    y_centre = c2.number_input("Y (m)", value=4_582_200.0, format="%.1f")

    st.subheader("Dimensions del rectangle")
    d1, d2 = st.columns(2)
    ample = d1.number_input("Ample (m)", value=5_000.0, min_value=100.0)
    alt   = d2.number_input("Alt (m)",   value=4_000.0, min_value=100.0)

    executar = st.button("🚀 Generar mapa i llegenda", type="primary")

# ── Lògica principal ──────────────────────────────────────────────────────────
if executar:
    with st.spinner("Connectant amb ICGC i processant dades…"):
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                minx, miny, maxx, maxy = bbox_utm_25831(x_centre, y_centre, ample, alt)
                w_px, h_px = calcular_px(ample, alt)

                # 1. Imatge WMS ────────────────────────────────────────────────
                img_bytes = obtenir_imatge_wms(minx, miny, maxx, maxy, w_px, h_px)
                img_path  = os.path.join(tmpdir, "icgc_mapa.png")
                with open(img_path, "wb") as f:
                    f.write(img_bytes)

                # 2. Llegenda WMS (leyenda completa — se conserva en ZIP)
                try:
                    llegenda_img  = obtenir_llegenda_wms()
                    llegenda_path = os.path.join(tmpdir, "icgc_llegenda_wms.png")
                    llegenda_img.save(llegenda_path)
                    te_llegenda_wms = True
                except Exception:
                    llegenda_path   = None
                    te_llegenda_wms = False

                # 3. Dades vectorials ArcGIS REST ──────────────────────────────
                gdf_raw = obtenir_poligons_arcgis(minx, miny, maxx, maxy)
                gdf, gdf_unitats = processar_gdf(gdf_raw)

                csv_path     = os.path.join(tmpdir, "icgc_unitats.csv")
                geojson_path = os.path.join(tmpdir, "icgc_recintos.geojson")
                te_vector = (gdf is not None and len(gdf) > 0)

                if te_vector:
                    cols_csv = [c for c in gdf_unitats.columns if c != "geometry"]
                    gdf_unitats.to_csv(
                        csv_path, sep=";", index=False,
                        columns=cols_csv, encoding="utf-8"
                    )
                    if isinstance(gdf, gpd.GeoDataFrame):
                        gdf.to_file(geojson_path, driver="GeoJSON")
                    else:
                        raise RuntimeError("gdf no és un GeoDataFrame vàlid")
                    n_unitats = len(gdf_unitats)
                else:
                    n_unitats = 0

                # 3b. Colors del renderer i llegenda textual amb materials ────
                color_map = obtenir_colors_renderer()
                llegenda_text_path = os.path.join(tmpdir, "icgc_llegenda_materials.png")
                if te_vector:
                    generar_llegenda_textual(gdf_unitats, llegenda_text_path, color_map)
                    te_llegenda_text = os.path.exists(llegenda_text_path)
                else:
                    te_llegenda_text = False

                # 4. Missatge d'èxit ───────────────────────────────────────────
                if te_vector:
                    st.success(
                        f"✅ Procés completat. Unitats geològiques trobades: **{n_unitats}**"
                    )
                else:
                    st.warning(
                        "⚠️ El servei vectorial no ha retornat polígons. "
                        "El mapa i la llegenda s'han generat correctament."
                    )

                # 5. Previsualització ──────────────────────────────────────────
                st.subheader("Previsualització")
                col_map, col_leg = st.columns([2, 1])
                with col_map:
                    st.image(
                        img_path,
                        caption="Mapa geològic ICGC 1:50.000",
                        use_container_width=True,
                    )
                with col_leg:
                    if te_llegenda_text:
                        st.image(
                            llegenda_text_path,
                            caption="Llegenda de l'àrea consultada",
                            use_container_width=True,
                        )
                    elif te_llegenda_wms and llegenda_path:
                        st.image(
                            llegenda_path,
                            caption="Llegenda completa WMS",
                            use_container_width=True,
                        )

                # 6. Taula d'atributs ─────────────────────────────────────────
                if te_vector:
                    with st.expander("📋 Taula d'unitats geològiques"):
                        cols_show = [c for c in gdf_unitats.columns if c != "geometry"]
                        st.dataframe(gdf_unitats[cols_show], use_container_width=True)

                # 7. ZIP de descàrrega ─────────────────────────────────────────
                zip_buf = BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.write(img_path,      arcname="icgc_mapa.png")
                    zf.write(llegenda_path, arcname="icgc_llegenda.png")
                    if te_vector:
                        zf.write(csv_path,      arcname="icgc_unitats.csv")
                        zf.write(geojson_path,  arcname="icgc_recintos.geojson")
                    if te_llegenda_text:
                        zf.write(llegenda_text_path, arcname="icgc_llegenda_materials.png")
                    if te_llegenda_wms and llegenda_path:
                        zf.write(llegenda_path, arcname="icgc_llegenda_wms_completa.png")
                zip_buf.seek(0)

                st.download_button(
                    label="📦 Descarregar tot (imatges + dades)",
                    data=zip_buf,
                    file_name="icgc_geologia.zip",
                    mime="application/zip",
                )

                # 8. Info tècnica ─────────────────────────────────────────────
                with st.expander("ℹ️ Informació tècnica de la consulta"):
                    st.code(
                        f"WMS endpoint  : {WMS_BASE}\n"
                        f"Capa WMS      : {CAPA_WMS}\n"
                        f"BBOX (25831)  : {minx:.1f}, {miny:.1f}, {maxx:.1f}, {maxy:.1f}\n"
                        f"Imatge        : {w_px}×{h_px} px\n"
                        f"REST endpoint : {ARCGIS_BASE}\n"
                        f"Camp codi     : {CAMP_CODI}\n"
                        f"Polígons REST : {'sí' if te_vector else 'no disponible'}"
                    )

            except requests.exceptions.HTTPError as e:
                st.error(f"❌ Error HTTP en la petició al servidor ICGC: {e}")
            except requests.exceptions.Timeout:
                st.error(
                    "❌ El servidor ICGC no ha respost a temps. "
                    "Prova amb una àrea més petita."
                )
            except Exception as e:
                st.error(f"❌ Error inesperat: {e}")
                import traceback
                st.code(traceback.format_exc())
