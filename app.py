# app.py
import os
from io import BytesIO

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from pipeline import (
    run_pipeline,
    COMPONENTS_CATALOG,
    append_to_gsheet,
)

load_dotenv()

st.set_page_config(
    page_title="Generador de instancias y rúbricas",
    layout="wide"
)

st.title("🧩 Generador de instancias verificadoras y rúbricas")
st.write(
    "A partir de un texto curricular (por ejemplo tu RAW_TEXT_MULTI), "
    "se generan instancias verificadoras, componentes y rúbricas por nivel."
)

# ----------------- Configuración Google Sheet de destino -----------------
# Esta es la hoja que compartiste:
# https://docs.google.com/spreadsheets/d/1wPejY3e41MwMtr5Jn0xPEX15i73h3kDH/edit?gid=844244215#gid=844244215
GSPREAD_SPREADSHEET_ID = "1wPejY3e41MwMtr5Jn0xPEX15i73h3kDH"
GSPREAD_WORKSHEET_GID = 844244215  # pestaña específica

st.caption(
    "Los resultados se acumularán en la pestaña de la Google Sheet configurada, "
    "además de poder descargar el Excel generado."
)

# ----------------- Sidebar: configuración -----------------
st.sidebar.header("Configuración")

# API KEY
default_key = os.getenv("OPENAI_API_KEY", "")
api_key = st.sidebar.text_input(
    "OPENAI_API_KEY",
    type="password",
    value=default_key,
    help="Tu clave de la API de OpenAI (puedes ponerla también en los secrets de Streamlit)."
)

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Área / materia
areas = sorted(COMPONENTS_CATALOG.keys())
default_area_index = 0
if "CIENCIAS SOCIALES" in areas:
    default_area_index = areas.index("CIENCIAS SOCIALES")

area = st.sidebar.selectbox(
    "Materia / Área",
    areas,
    index=default_area_index
)

# Grado
grado_num = st.sidebar.number_input(
    "Grado (número)",
    min_value=-2,
    max_value=11,
    value=5,
    step=1,
    help="Convención: -2=Prejardín, -1=Jardín, 0=Transición, 1-11 básicos."
)

# Instancias por desempeño
ivs_per_des = st.sidebar.number_input(
    "Instancias por desempeño",
    min_value=1,
    max_value=10,
    value=2,
    step=1,
    help="Cantidad FINAL de instancias verificadoras por desempeño."
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Al presionar **Generar**, se llama a la API de OpenAI para parsear el texto, "
    "generar instancias, clasificar componentes y construir rúbricas."
)

# ----------------- Entrada de texto -----------------
st.subheader("1. Texto fuente (RAW_TEXT_MULTI)")

tab_text, tab_file = st.tabs(["Pegar texto", "Subir .txt"])

raw_text = ""

with tab_text:
    raw_text = st.text_area(
        "Pega aquí el texto curricular (por ejemplo el RAW_TEXT_MULTI entero).",
        height=400,
        placeholder="Pega aquí el texto con GRADO, COMPONENTS, Unit 1..., etc."
    )

with tab_file:
    uploaded = st.file_uploader("O sube un archivo .txt con el contenido", type=["txt"])
    if uploaded is not None:
        raw_text = uploaded.read().decode("utf-8", errors="ignore")

# ----------------- Botón de ejecución -----------------
st.subheader("2. Procesar")

if st.button("🚀 Generar instancias y rúbricas"):
    if not api_key:
        st.error("Debes ingresar tu OPENAI_API_KEY en la barra lateral o en las variables de entorno.")
    elif not raw_text or not raw_text.strip():
        st.error("Debes pegar o subir algún texto.")
    else:
        with st.spinner("Procesando texto y generando rúbricas..."):
            try:
                df = run_pipeline(
                    raw_text=raw_text,
                    area=area,
                    grado_num=int(grado_num),
                    ivs_per_desempeno=int(ivs_per_des)
                )
            except Exception as e:
                st.error(f"Ocurrió un error al ejecutar el pipeline: {e}")
            else:
                st.success(f"Se generaron {len(df)} filas.")

                # ---------- Actualizar Google Sheet ----------
                try:
                    total_gs = append_to_gsheet(
                        df_new=df,
                        spreadsheet_id=GSPREAD_SPREADSHEET_ID,
                        worksheet_gid=GSPREAD_WORKSHEET_GID,
                    )
                    st.success(
                        f"✅ Google Sheet actualizada correctamente.\n\n"
                        f"ID: `{GSPREAD_SPREADSHEET_ID}`, gid: `{GSPREAD_WORKSHEET_GID}`\n"
                        f"Filas totales ahora en esa pestaña: {total_gs}"
                    )
                except Exception as e:
                    st.error(f"⚠️ No se pudo actualizar la Google Sheet: {e}")

                # ---------- Vista previa ----------
                st.subheader("3. Vista previa de resultados")
                st.dataframe(df, use_container_width=True)

                # ---------- Excel para descargar ----------
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df.to_excel(writer, index=False, sheet_name="Instancias")
                buffer.seek(0)

                st.download_button(
                    label="💾 Descargar Excel con instancias y rúbricas",
                    data=buffer,
                    file_name="instancias_verificadoras.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
