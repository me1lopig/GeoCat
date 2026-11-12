# 🗺️ GeoCat

**Generador de cartografía geológica de Cataluña con datos del Institut Cartogràfic i Geològic de Catalunya (ICGC).**

GeoCat permite seleccionar un área mediante coordenadas UTM y obtener su mapa geológico, una leyenda con las unidades presentes y, cuando el servicio lo permite, datos vectoriales para su uso en herramientas SIG.

## ✨ Funcionalidades

- 🧭 Selección del área mediante coordenadas **ETRS89 / UTM zona 31N (EPSG:25831)** y dimensiones en metros.
- 🪨 Descarga de cartografía geológica del ICGC a escala **1:50.000**.
- 🛰️ Capas de fondo opcionales: ortofoto, mapa topográfico, mapa topográfico gris o solo geología.
- 🎨 Ajuste de la opacidad de la capa geológica y composición con el mapa de fondo.
- 🔎 Identificación de unidades geológicas mediante la API ArcGIS REST; si no está disponible, utiliza OCR y análisis de colores como alternativa.
- 📋 Consulta de códigos, descripciones y edades geológicas en una tabla.
- 📦 Descarga de resultados en un archivo ZIP: mapa PNG, leyenda y, si están disponibles, archivos CSV y GeoJSON.

## 📁 Estructura del repositorio

Este repositorio contiene los scripts de la aplicación y sus dependencias en la raíz:

| Archivo | Descripción |
| --- | --- |
| `geode_icgc_4.py` | Versión más completa: consulta directa de unidades, alternativa OCR/colores, capas de fondo actualizables y descarga de datos vectoriales. |
| `geode_icgc_3.py` | Versión con consulta directa de unidades y alternativa OCR/colores. |
| `geode_icgc_2.py` | Versión con capas de fondo, composición del mapa e identificación de unidades. |
| `geode_icgc_1.py` | Versión con identificación mediante OCR y colores dominantes. |
| `geode_icgc.py` | Versión inicial del generador de cartografía y leyenda. |
| `requeriments.txt` | Lista de dependencias de Python necesarias para la aplicación. |
| `LICENSE` | Licencia del proyecto. |
| `README.md` | Documentación e instrucciones de uso. |

> 🚀 Para empezar, se recomienda ejecutar `geode_icgc_4.py`.

## 🛠️ Requisitos

- Python 3.10 o superior.
- Conexión a Internet para consultar los servicios cartográficos del ICGC.
- Tesseract OCR es opcional; se utiliza como parte del método alternativo si la consulta directa de unidades no está disponible.

## 🚀 Instalación y ejecución

1. Clona el repositorio y entra en su directorio:

   ```bash
   git clone https://github.com/me1lopig/GeoCat.git
   cd GeoCat
   ```

2. (Recomendado) Crea y activa un entorno virtual:

   ```bash
   python -m venv .venv
   ```

   En Linux o macOS:

   ```bash
   source .venv/bin/activate
   ```

   En Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Instala las dependencias:

   ```bash
   pip install -r requeriments.txt
   ```

4. Inicia la aplicación:

   ```bash
   streamlit run geode_icgc_4.py
   ```

Streamlit abrirá la aplicación en el navegador. Si deseas probar otra versión, sustituye el nombre del script en el comando.

### 🔤 Activar OCR (opcional)

Además del paquete Python `pytesseract`, el método OCR necesita que el ejecutable de Tesseract esté instalado en el sistema. Por ejemplo, en Debian o Ubuntu:

```bash
sudo apt-get install tesseract-ocr
```

## 🧑‍💻 Uso

1. Introduce las coordenadas del centro del área y su anchura y altura en metros.
2. Elige la capa de fondo y la opacidad de la geología.
3. Pulsa **Generar mapa y leyenda**.
4. Consulta la vista previa, las unidades identificadas y descarga los resultados con **Descargar todo**.

📍 Las coordenadas deben corresponder a **ETRS89 / UTM zona 31N (EPSG:25831)**. El resultado depende de la disponibilidad de los servicios del ICGC.

## 📚 Tecnologías y datos

- 🐍 **Python** y **Streamlit** para la aplicación.
- 🌐 **WMS** para obtener mapas y capas de fondo.
- 🗺️ **ArcGIS REST** para consultar unidades y datos geológicos.
- 🧮 **NumPy**, **Pandas**, **Pillow**, **GeoPandas** y **Shapely** para el procesamiento de imágenes y datos espaciales.
- 🔤 **pytesseract** para el reconocimiento OCR opcional.

Los mapas y datos geológicos se consultan desde los servicios del **ICGC**; se requiere conexión a Internet.

## 📄 Licencia

Consulta el archivo [`LICENSE`](LICENSE) para conocer los términos de uso.
