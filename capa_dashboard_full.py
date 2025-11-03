import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import numpy as np
# ==========================
# CONFIG
# ==========================
st.set_page_config(page_title="CAPA Dashboard", layout="wide")

st.markdown("""
    <style>
    h1 {font-size: 28px !important;}
    h2 {font-size: 20px !important;}
    h3 {font-size: 16px !important;}
    </style>
    """, unsafe_allow_html=True)

TODAY = datetime.today()

# ==========================
# LOAD DATA
# ==========================
@st.cache_data(ttl=60)
def load_data():
    # Leer hoja principal (el archivo debe estar en el mismo repositorio que el script)
    EXCEL_FILE = "CAPA_follow up (11).xlsx"

    df = pd.read_excel(EXCEL_FILE, sheet_name="DataBase")
    df.columns = df.columns.str.strip().str.lower()
    if "responsible site" in df.columns:
        df["responsible site"] = df["responsible site"].astype(str).str.strip()

    for col in ["created date", "current phase due date", "closed date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Leer hoja "Committed Date"
    try:
        df_committed = pd.read_excel(EXCEL_FILE, sheet_name="Committed Date")
        df_committed.columns = df_committed.columns.str.strip().str.lower()

        if "capa number" in df_committed.columns:
            cols_to_merge = ["capa number", "committed date"]

            if "to be overdue" in df_committed.columns:
                cols_to_merge.append("to be overdue")
                committed_lookup = df_committed[cols_to_merge].copy()
                committed_lookup = committed_lookup.rename(columns={"to be overdue": "at risk"})
            elif "at risk" in df_committed.columns:
                cols_to_merge.append("at risk")
                committed_lookup = df_committed[cols_to_merge].copy()
            else:
                committed_lookup = df_committed[["capa number", "committed date"]].copy()

            committed_lookup = committed_lookup.drop_duplicates(subset=["capa number"], keep="last")
            df = df.merge(committed_lookup, on="capa number", how="left", suffixes=("", "_committed"))

            if "committed date_committed" in df.columns:
                df["committed date"] = df["committed date_committed"]
                df = df.drop(columns=["committed date_committed"])
            if "at risk_committed" in df.columns:
                df["at risk"] = df["at risk_committed"]
                df = df.drop(columns=["at risk_committed"])
    except Exception as e:
        st.warning(f"No se pudo leer la hoja 'Committed Date': {e}")

    # Calcular extensiones de fecha de vencimiento
    try:
        if "step id" in df.columns and "sign-off date" in df.columns and "current phase" in df.columns:
            df_ext = df[df["step id"].str.lower().str.strip() == "tapproveduedateextension"].copy()
            extensions_count = {}

            for capa_num in df["capa number"].unique():
                if pd.isna(capa_num):
                    continue

                capa_rows = df[df["capa number"] == capa_num].copy()
                current_phase = capa_rows["current phase"].iloc[-1] if len(capa_rows) > 0 else None

                if pd.isna(current_phase):
                    extensions_count[capa_num] = 0
                    continue

                phase_entry_rows = capa_rows[capa_rows["current phase"] == current_phase]
                if len(phase_entry_rows) == 0:
                    extensions_count[capa_num] = 0
                    continue

                phase_entry_date = pd.to_datetime(phase_entry_rows["sign-off date"].iloc[0], errors="coerce")
                capa_extensions = df_ext[df_ext["capa number"] == capa_num].copy()
                capa_extensions["sign-off date"] = pd.to_datetime(capa_extensions["sign-off date"], errors="coerce").dt.date

                if pd.notna(phase_entry_date):
                    capa_extensions = capa_extensions[
                        pd.to_datetime(capa_extensions["sign-off date"]) >= phase_entry_date
                    ]

                extensions_count[capa_num] = capa_extensions["sign-off date"].nunique()

            df["due_date_extensions"] = df["capa number"].map(extensions_count).fillna(0).astype(int)
        else:
            df["due_date_extensions"] = 0
    except Exception as e:
        st.warning(f"No se pudo calcular extensiones: {e}")
        df["due_date_extensions"] = 0

    return df

df = load_data()

st.markdown("<h1>CAPA DASHBOARD</h1>", unsafe_allow_html=True)

# ==========================
# SECTION: Indicators
# ==========================

st.markdown("<h1>Indicators</h1>", unsafe_allow_html=True)

TODAY_DATE = date.today()

df_clean = df[df["capa number"].notna()].copy()
df_clean["created date"] = pd.to_datetime(df_clean["created date"], errors="coerce")
df_clean["closed date"] = pd.to_datetime(df_clean["closed date"], errors="coerce")
df_clean["age_days"] = (TODAY - df_clean["created date"]).dt.days

# ------------------ General Numbers ------------------
total_global = df_clean["capa number"].nunique()
inworks_global = df_clean.loc[df_clean["primary status"].str.lower() != "closed", "capa number"].nunique()
closed_global = df_clean.loc[df_clean["primary status"].str.lower() == "closed", "capa number"].nunique()

total_1100 = df_clean.loc[df_clean["responsible site"] == "1100", "capa number"].nunique()
inworks_1100 = df_clean.loc[(df_clean["responsible site"] == "1100") & (df_clean["primary status"].str.lower() != "closed"), "capa number"].nunique()
closed_1100 = df_clean.loc[(df_clean["responsible site"] == "1100") & (df_clean["primary status"].str.lower() == "closed"), "capa number"].nunique()

col1, col2 = st.columns(2)
with col1:
    st.markdown("<h2>General Numbers - Global</h2>", unsafe_allow_html=True)
    st.table(pd.DataFrame({"Item": ["Total CAPAs", "Total In works", "Total Closed"],
                           "Number": [int(total_global), int(inworks_global), int(closed_global)]}))
with col2:
    st.markdown("<h2>General Numbers - Site 1100</h2>", unsafe_allow_html=True)
    st.table(pd.DataFrame({"Item": ["Total CAPAs", "Total In works", "Total Closed"],
                           "Number": [int(total_1100), int(inworks_1100), int(closed_1100)]}))

# ------------------ CAPAs by Phase ------------------
def map_phase(p):
    p = str(p).lower().strip()
    if p in ["investigation", "investigation approval", "initiation verification"]:
        return "Investigation"
    if p in ["implementation", "implementation verification"]:
        return "Implementation"
    if p in ["completion review", "effectiveness review", "effectiveness monitoring"]:
        return "Effectiveness Monitoring"
    return None

df_inworks = df_clean[df_clean["primary status"].str.lower() != "closed"].copy()
df_inworks["phase_grouped"] = df_inworks["phase"].apply(map_phase)
df_inworks_tbl = df_inworks[df_inworks["phase_grouped"].notna()].copy()

# Tablas de Phase (Number + Age)
phase_global_tbl = df_inworks_tbl.groupby("phase_grouped").agg(
    Number=("capa number", "nunique"),
    Age=("age_days", "mean")
).reset_index().rename(columns={"phase_grouped": "Phase"})
phase_global_tbl["Age (days)"] = phase_global_tbl["Age"].round(0).astype("Int64")
phase_global_tbl["Number"] = phase_global_tbl["Number"].astype(int)
phase_global_tbl = phase_global_tbl[["Phase", "Number", "Age (days)"]]

df_inworks_1100_tbl = df_inworks_tbl[df_inworks_tbl["responsible site"] == "1100"].copy()
phase_1100_tbl = df_inworks_1100_tbl.groupby("phase_grouped").agg(
    Number=("capa number", "nunique"),
    Age=("age_days", "mean")
).reset_index().rename(columns={"phase_grouped": "Phase"})
phase_1100_tbl["Age (days)"] = phase_1100_tbl["Age"].round(0).astype("Int64")
phase_1100_tbl["Number"] = phase_1100_tbl["Number"].astype(int)
phase_1100_tbl = phase_1100_tbl[["Phase", "Number", "Age (days)"]]

# Cálculo de overdue para coloración
df_inworks_due = df_inworks.copy()
df_inworks_due["current phase due date"] = pd.to_datetime(df_inworks_due["current phase due date"], errors="coerce")

# IMPORTANTE: Agregamos las columnas committed date, at risk, owner y coordinator
agg_dict = {
    "phase": "last",
    "responsible site": "last",
    "current phase due date": "max"
}

# Agregar columnas opcionales si existen
if "committed date" in df_inworks_due.columns:
    agg_dict["committed date"] = "last"
if "at risk" in df_inworks_due.columns:
    agg_dict["at risk"] = "last"
if "due_date_extensions" in df_inworks_due.columns:
    agg_dict["due_date_extensions"] = "max"
if "capa owner name" in df_inworks_due.columns:
    agg_dict["capa owner name"] = "last"
if "capa coordinator name" in df_inworks_due.columns:
    agg_dict["capa coordinator name"] = "last"
if "title" in df_inworks_due.columns:
    agg_dict["title"] = "last"

capa_inworks = df_inworks_due.groupby("capa number").agg(agg_dict).reset_index()

capa_inworks["Phase"] = capa_inworks["phase"].apply(map_phase)
capa_inworks = capa_inworks[capa_inworks["Phase"].notna()].copy()
capa_inworks["due_date"] = capa_inworks["current phase due date"].dt.date
capa_inworks["status_due"] = capa_inworks["due_date"].apply(lambda d: "Overdue" if (pd.notna(d) and d < TODAY_DATE) else "On time")

overdue_global = capa_inworks[capa_inworks["status_due"] == "Overdue"].groupby("Phase")["capa number"].nunique().to_dict()
overdue_1100 = capa_inworks[(capa_inworks["responsible site"] == "1100") & (capa_inworks["status_due"] == "Overdue")] \
                           .groupby("Phase")["capa number"].nunique().to_dict()

thr_global = {"Investigation": 4, "Implementation": 6, "Effectiveness Monitoring": 4}
thr_1100  = {"Investigation": 2, "Implementation": 3, "Effectiveness Monitoring": 2}

def style_phase_numbers(df_table, overdue_counts, thresholds):
    def _row_style(row):
        phase = row["Phase"]
        od = int(overdue_counts.get(phase, 0))
        limit = thresholds.get(phase, 9999)
        
        if od > limit:
            num_style = "color: red; font-weight: bold"
        elif od == 0:
            num_style = "color: green; font-weight: bold"
        else:
            num_style = ""
        
        return ["", num_style, ""]
    
    return df_table.style.apply(lambda r: _row_style(r), axis=1)

# Mostrar tablas de Phase
col3, col4 = st.columns(2)
with col3:
    st.markdown("<h2>CAPAs by Phase - Global</h2>", unsafe_allow_html=True)
    st.dataframe(style_phase_numbers(phase_global_tbl.copy(), overdue_global, thr_global))
    
with col4:
    st.markdown("<h2>CAPAs by Phase - Site 1100</h2>", unsafe_allow_html=True)
    st.dataframe(style_phase_numbers(phase_1100_tbl.copy(), overdue_1100, thr_1100))

# ------------------ Gráficos On time vs Overdue ------------------
def make_bar_chart_totals_only(df, title):
    phase_status = df.groupby(["Phase","Status"]).agg(CAPAs=("capa number","nunique")).reset_index()
    phase_status = phase_status.pivot(index="Phase", columns="Status", values="CAPAs").fillna(0)
    for col in ["On time", "Overdue"]:
        if col not in phase_status.columns:
            phase_status[col] = 0
    phase_status = phase_status.reindex(["Investigation","Implementation","Effectiveness Monitoring"]).fillna(0)

    phases = phase_status.index.tolist()
    on_time = phase_status["On time"].astype(int).tolist()
    overdue = phase_status["Overdue"].astype(int).tolist()
    totals = np.array(on_time) + np.array(overdue)

    fig = go.Figure()
    fig.add_bar(
        x=phases, y=on_time, name="On time", marker_color="green",
        text=on_time, textposition="inside", insidetextanchor="middle",
        hovertemplate="Phase: %{x}<br>On time: %{y}<extra></extra>"
    )
    fig.add_bar(
        x=phases, y=overdue, name="Overdue", marker_color="red",
        text=overdue, textposition="inside", insidetextanchor="middle",
        hovertemplate="Phase: %{x}<br>Overdue: %{y}<extra></extra>"
    )

    for i, total in enumerate(totals):
        fig.add_annotation(
            x=phases[i], y=totals[i] + max(totals)*0.05, text=str(total),
            showarrow=False, yshift=5, font=dict(size=11)
        )

    fig.update_layout(
        barmode="stack", 
        title=dict(text=title, font=dict(size=18)),
        legend=dict(orientation="h", y=-0.2),
        uniformtext_minsize=8, uniformtext_mode="hide"
    )
    return fig, phase_status

capa_inworks["Current Due Date"] = pd.to_datetime(capa_inworks["current phase due date"], errors="coerce").dt.date
capa_inworks["Status"] = capa_inworks["Current Due Date"].apply(lambda d: "Overdue" if (pd.notna(d) and d < TODAY_DATE) else "On time")

fig_global, phase_global = make_bar_chart_totals_only(capa_inworks, "Global - In Works (On time vs Overdue)")
site_1100 = capa_inworks[capa_inworks["responsible site"] == "1100"].copy()
fig_1100, phase_1100 = make_bar_chart_totals_only(site_1100, "Site 1100 - In Works (On time vs Overdue)")

col5, col6 = st.columns(2)
with col5:
    st.plotly_chart(fig_global, use_container_width=True, key="fig_global_ontime_overdue")
with col6:
    st.plotly_chart(fig_1100, use_container_width=True, key="fig_1100_ontime_overdue")

# ==========================================================
# SECTION: CAPAs > 2 years - Table + Bar Chart with Totals
# ==========================================================

st.markdown("<h1>CAPAs older than 2 years</h1>", unsafe_allow_html=True)

def age_brackets(df_src):
    """Return a table with CAPAs counts by age brackets (IN WORKS only)."""
    age_data = {
        "Item": ["Older than 2 years", "Older than 1 year"],
        "Number": [
            int(df_src.loc[df_src["age_days"] > 730, "capa number"].nunique()),
            int(df_src.loc[df_src["age_days"] > 365, "capa number"].nunique())
        ]
    }
    return pd.DataFrame(age_data)

def bar_chart_over2yrs(df_src):
    """Bar chart for CAPAs > 2 years by phase and status, with totals."""
    df = df_src.copy()
    df["created date"] = pd.to_datetime(df["created date"], errors="coerce")
    df["current phase due date"] = pd.to_datetime(df["current phase due date"], errors="coerce")
    df["age_days"] = (TODAY - df["created date"]).dt.days
    df["phase_grouped"] = df["phase"].apply(map_phase)
    df = df[(df["age_days"] > 730) &
            (df["phase_grouped"].isin(["Implementation", "Effectiveness Monitoring"]))]

    df["Status"] = df["current phase due date"].apply(
        lambda d: "Overdue" if (pd.notna(d) and d.date() < TODAY_DATE) else "On time"
    )

    summary = (
        df.groupby(["phase_grouped", "Status"])["capa number"]
        .nunique()
        .reset_index()
        .pivot(index="phase_grouped", columns="Status", values="capa number")
        .fillna(0)
        .reset_index()
    )

    # Asegurar orden: Implementation primero
    summary["phase_grouped"] = pd.Categorical(
        summary["phase_grouped"], ["Implementation", "Effectiveness Monitoring"]
    )
    summary = summary.sort_values("phase_grouped")

    summary["Total"] = summary["On time"] + summary["Overdue"]

    fig = go.Figure()

    # Barras apiladas
    fig.add_bar(
        x=summary["phase_grouped"],
        y=summary["On time"],
        name="On time",
        marker_color="#2ca02c",
        text=summary["On time"],
        textposition="inside"
    )
    fig.add_bar(
        x=summary["phase_grouped"],
        y=summary["Overdue"],
        name="Overdue",
        marker_color="#d62728",
        text=summary["Overdue"],
        textposition="inside"
    )

    # Totales sobre las barras
    for i, phase in enumerate(summary["phase_grouped"]):
        total = summary.loc[summary["phase_grouped"] == phase, "Total"].values[0]
        fig.add_annotation(
            x=phase,
            y=total + 0.3,
            text=f"<b>{int(total)}</b>",
            showarrow=False,
            font=dict(color="white", size=14)
        )

    fig.update_layout(
        barmode="stack",
        height=400,
        xaxis_title="Phase",
        yaxis_title="Number of CAPAs > 2 years",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=60, b=40, l=0, r=0)
    )
    return fig

# ---------- Layout ----------
colA, colB = st.columns(2)

with colA:
    st.markdown("<h2 style='text-align:center;'>Global</h2>", unsafe_allow_html=True)
    st.table(age_brackets(df_inworks))
    fig_global_2yrs = bar_chart_over2yrs(df_inworks)
    st.plotly_chart(fig_global_2yrs, use_container_width=True, key="fig_global_over2yrs")

with colB:
    st.markdown("<h2 style='text-align:center;'>Site 1100</h2>", unsafe_allow_html=True)
    df_1100 = df_inworks[df_inworks["responsible site"] == "1100"]
    st.table(age_brackets(df_1100))
    fig_1100_2yrs = bar_chart_over2yrs(df_1100)
    st.plotly_chart(fig_1100_2yrs, use_container_width=True, key="fig_1100_over2yrs")

# ==========================
# Overdue CAPAs Table
# ==========================
st.markdown("<h2>Overdue CAPAs</h2>", unsafe_allow_html=True)

overdue_all = capa_inworks[capa_inworks["Status"] == "Overdue"].copy()

# Solo mostrar estas columnas (SIN At Risk, CON Owner y Coordinator)
columns_to_show = ["capa number", "responsible site", "Phase", "Current Due Date"]

# Agregar committed date si existe
if "committed date" in overdue_all.columns:
    columns_to_show.append("committed date")
# Agregar owner y coordinator
if "capa owner name" in overdue_all.columns:
    columns_to_show.append("capa owner name")
if "capa coordinator name" in overdue_all.columns:
    columns_to_show.append("capa coordinator name")

# Seleccionar solo esas columnas
overdue_all = overdue_all[columns_to_show]

# Renombrar para que se vea bien
rename_dict = {
    "capa number": "CAPA Number",
    "responsible site": "Responsible Site"
}
if "committed date" in overdue_all.columns:
    rename_dict["committed date"] = "Committed Date"
if "capa owner name" in overdue_all.columns:
    rename_dict["capa owner name"] = "CAPA Owner"
if "capa coordinator name" in overdue_all.columns:
    rename_dict["capa coordinator name"] = "CAPA Coordinator"

overdue_all = overdue_all.rename(columns=rename_dict)

# Formatear Current Due Date (que sí es fecha)
overdue_all["Current Due Date"] = pd.to_datetime(overdue_all["Current Due Date"], errors="coerce").dt.strftime("%d-%b-%Y")

# Formatear Committed Date: si es fecha convertir, si es texto dejar como está, si está vacío poner TBD
if "Committed Date" in overdue_all.columns:
    def format_committed_date(val):
        if pd.isna(val) or val == "" or str(val).strip() == "":
            return "TBD"
        try:
            date_val = pd.to_datetime(val, errors="coerce")
            if pd.notna(date_val):
                return date_val.strftime("%d-%b-%Y")
            else:
                return val
        except:
            return val
    
    overdue_all["Committed Date"] = overdue_all["Committed Date"].apply(format_committed_date)

# Ordenar por fase y fecha
phase_order = {"Investigation": 1, "Implementation": 2, "Effectiveness Monitoring": 3}
overdue_all["PhaseOrder"] = overdue_all["Phase"].map(phase_order)
overdue_all = overdue_all.sort_values(by=["PhaseOrder", "Current Due Date"], ascending=[True, True])
overdue_all = overdue_all.drop(columns="PhaseOrder").reset_index(drop=True)

st.dataframe(overdue_all)

# ==========================
# CAPAs Due Next Month - Site 1100 Only
# ==========================
st.markdown("<h2>CAPAs due in the next 2 months</h2>", unsafe_allow_html=True)

# Calcular el rango del próximo mes
next_month_start = TODAY_DATE + timedelta(days=1)
next_month_end = TODAY_DATE + timedelta(days=60)

# Filtrar CAPAs que vencen en el próximo mes (que no estén overdue) Y que sean del sitio 1100
next_month_capas = capa_inworks[
    (capa_inworks["Status"] == "On time") &
    (capa_inworks["Current Due Date"] >= next_month_start) &
    (capa_inworks["Current Due Date"] <= next_month_end) &
    (capa_inworks["responsible site"] == "1100")
].copy()

# Columnas a mostrar en orden: CAPA Number, Title, Phase, Owner, Coordinator, Current Due Date, To Be Overdue, Extensions
columns_next_month = ["capa number"]

if "title" in next_month_capas.columns:
    columns_next_month.append("title")

columns_next_month.extend(["responsible site", "Phase", "Current Due Date"])

if "capa owner name" in next_month_capas.columns:
    columns_next_month.append("capa owner name")
if "capa coordinator name" in next_month_capas.columns:
    columns_next_month.append("capa coordinator name")
if "at risk" in next_month_capas.columns:
    columns_next_month.append("at risk")
if "due_date_extensions" in next_month_capas.columns:
    columns_next_month.append("due_date_extensions")

next_month_capas = next_month_capas[columns_next_month]

# Poner valores por defecto ANTES de renombrar
if "at risk" in next_month_capas.columns:
    def format_to_be_overdue(val):
        if pd.isna(val) or val == "" or str(val).strip() == "" or val is None or str(val).lower() == "none":
            return "On Time"
        return val
    next_month_capas["at risk"] = next_month_capas["at risk"].apply(format_to_be_overdue)

# Renombrar columnas
rename_dict_next = {
    "capa number": "CAPA Number",
    "responsible site": "Responsible Site"
}
if "title" in next_month_capas.columns:
    rename_dict_next["title"] = "Title"
if "capa owner name" in next_month_capas.columns:
    rename_dict_next["capa owner name"] = "CAPA Owner"
if "capa coordinator name" in next_month_capas.columns:
    rename_dict_next["capa coordinator name"] = "CAPA Coordinator"
if "at risk" in next_month_capas.columns:
    rename_dict_next["at risk"] = "To Be Overdue"
if "due_date_extensions" in next_month_capas.columns:
    rename_dict_next["due_date_extensions"] = "Due Date Extensions"

next_month_capas = next_month_capas.rename(columns=rename_dict_next)

# Formatear fecha
next_month_capas["Current Due Date"] = pd.to_datetime(next_month_capas["Current Due Date"], errors="coerce").dt.strftime("%d-%b-%Y")

# Poner "On Time" en To Be Overdue cuando esté vacío o None
if "To Be Overdue" in next_month_capas.columns:
    def format_at_risk(val):
        if pd.isna(val) or val == "" or str(val).strip() == "" or val is None or str(val).lower() == "none":
            return "On Time"
        return val
    
    next_month_capas["To Be Overdue"] = next_month_capas["To Be Overdue"].apply(format_at_risk)

# Ordenar por fase y fecha
next_month_capas["PhaseOrder"] = next_month_capas["Phase"].map(phase_order)
next_month_capas = next_month_capas.sort_values(by=["PhaseOrder", "Current Due Date"], ascending=[True, True])
next_month_capas = next_month_capas.drop(columns="PhaseOrder").reset_index(drop=True)

# Aplicar estilo condicional a la columna To Be Overdue
def highlight_to_be_overdue(row):
    styles = [''] * len(row)
    if "To Be Overdue" in next_month_capas.columns:
        col_idx = list(next_month_capas.columns).index("To Be Overdue")
        val = str(row["To Be Overdue"]).lower()
        if "risk" in val or "to be overdue" in val:
            styles[col_idx] = 'color: red; font-weight: bold'
    return styles

st.dataframe(next_month_capas.style.apply(highlight_to_be_overdue, axis=1), use_container_width=True)

# ==========================
# Recovery Overview Section
# ==========================
st.markdown("<h1>Recovery Overview</h1>", unsafe_allow_html=True)

# Semanas (extender hasta 55)
weeks = [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
weeks_real = [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
weeks_prediction = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]  # Conexión continua

# Backlog inicial
initial_backlog = 11

# CAPAs recuperadas por semana (no acumulado) – valores reales hasta S45
recovered_per_week = [2, 1, 1, 1, 0, 1, 2, 0, 0, 0, 0, 0, 0]

# Nuevas CAPAs overdue que aparecen por semana – valores reales hasta S45
new_overdue_per_week = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Calcular backlog semana a semana – hasta S45
backlog = [initial_backlog]
for i in range(len(weeks_real) - 1):
    next_backlog = backlog[-1] + new_overdue_per_week[i] - recovered_per_week[i]
    backlog.append(next_backlog)

# Valores reales confirmados sitio 1100 (S42–S45)
backlog[-4:] = [3, 3, 3, 5]

# Total Recovery acumulado (hasta S45)
total_recovery = []
cumulative = 0
for i in range(len(weeks_real)):
    cumulative += recovered_per_week[i]
    total_recovery.append(cumulative)
total_recovery[-4:] = [8, 8, 8, 6]

# Recovery Rate basado en backlog inicial (hasta S45)
recovery_rate = [round((rec / initial_backlog) * 100) for rec in total_recovery]
recovery_rate[-4:] = [73, 73, 73, 55]

# === Production Line (S45–S55) ===
def get_week_number(date_val):
    if pd.isna(date_val):
        return None
    return date_val.isocalendar()[1]

try:
    capa_inworks_1100 = capa_inworks[capa_inworks["responsible site"] == "1100"].copy()
    overdue_capas = capa_inworks_1100[capa_inworks_1100["Status"] == "Overdue"].copy()
    production_recovered = [0] * len(weeks_prediction)
    production_new_overdue = [0] * len(weeks_prediction)
    phases_to_be_overdue = {week: {"Investigation": 0, "Implementation": 0, "Effectiveness Monitoring": 0}
                            for week in weeks_prediction}

    # Recuperadas (committed date)
    for _, row in overdue_capas.iterrows():
        committed = row.get("committed date", None)
        if pd.notna(committed) and str(committed).strip() != "" and str(committed).upper() != "TBD":
            committed_date = pd.to_datetime(committed, errors="coerce")
            if pd.notna(committed_date):
                wk = get_week_number(committed_date)
                if wk in weeks_prediction:
                    idx = weeks_prediction.index(wk)
                    production_recovered[idx] += 1

    # Nuevas overdue (to be overdue)
    capas_to_be_overdue = capa_inworks_1100[
        (capa_inworks_1100["Status"] == "On time") &
        (capa_inworks_1100.get("at risk", "").astype(str)
         .str.lower().str.contains("risk|to be overdue", case=False, na=False))
    ].copy()
    for _, row in capas_to_be_overdue.iterrows():
        due_date = row.get("Current Due Date", None)
        phase = row.get("Phase", "Unknown")
        if pd.notna(due_date):
            due_datetime = pd.to_datetime(due_date, errors="coerce")
            if pd.notna(due_datetime):
                wk = get_week_number(due_datetime)
                if wk in weeks_prediction:
                    idx = weeks_prediction.index(wk)
                    production_new_overdue[idx] += 1
                    if phase in phases_to_be_overdue[wk]:
                        phases_to_be_overdue[wk][phase] += 1

    # Backlog proyectado: backlog_t = backlog_(t-1) + nuevas_overdue_t - recuperadas_t
    production_line = [backlog[-1]]
    for i in range(1, len(weeks_prediction)):
        next_prod = production_line[-1] + production_new_overdue[i] - production_recovered[i]
        production_line.append(next_prod)

    # Total Recovery proyectado
    total_recovery_projection = [total_recovery[-1]]
    recovery_rate_projection = [recovery_rate[-1]]
    cumulative_projected = total_recovery[-1]
    for i in range(1, len(weeks_prediction)):
        cumulative_projected += production_recovered[i] - production_new_overdue[i]
        total_recovery_projection.append(cumulative_projected)
        rate = round((cumulative_projected / initial_backlog) * 100)
        recovery_rate_projection.append(max(0, min(100, rate)))

except Exception as e:
    st.warning(f"No se pudo calcular Production Line: {e}")
    production_line = [backlog[-1]] * len(weeks_prediction)
    recovery_rate_projection = [recovery_rate[-1]] * len(weeks_prediction)

# Crear DataFrame
recovery_data = {"": ["Backlog", "Total Recovery", "Recovery Rate", "Prediction Line"]}
for i, week in enumerate(weeks_real + weeks_prediction):
    if week in weeks_real:
        idx_real = weeks_real.index(week)
        recovery_data[str(week)] = [backlog[idx_real], total_recovery[idx_real],
                                    f"{recovery_rate[idx_real]}%", "-"]
    else:
        idx_pred = weeks_prediction.index(week)
        recovery_data[str(week)] = ["-", "-", "-", production_line[idx_pred]]
df_recovery = pd.DataFrame(recovery_data)
st.dataframe(df_recovery, use_container_width=True)

# === Gráfico ===
fig = go.Figure()

# Línea roja: Backlog (33–45)
fig.add_trace(go.Scatter(
    x=weeks_real, y=backlog,
    mode="lines+markers+text",
    name="Backlog",
    text=backlog, textposition="top center",
    line=dict(color="red", width=2),
    hovertemplate="Week %{x}<br>Backlog: %{y}<extra></extra>"
))

# Línea naranja: Prediction Line (45–55) – conexión continua
fig.add_trace(go.Scatter(
    x=weeks_prediction, y=production_line,
    mode="lines+markers+text",
    name="Prediction Line",
    text=[str(v) for v in production_line],
    textposition="bottom center",
    line=dict(color="orange", width=2, dash="dash"),
    hovertemplate="Week %{x}<br>Prediction: %{y}<extra></extra>"
))

# === KPI Break (idéntico al original) ===
impl_limit_1100 = thr_1100.get("Implementation", 9999)
impl_base_1100 = int(overdue_1100.get("Implementation", 0))
cum_rec, cum_new_impl = 0, 0
for i, wk in enumerate(weeks_prediction):
    if i > 0:
        cum_rec += production_recovered[i]
        cum_new_impl += phases_to_be_overdue[wk]["Implementation"]
    impl_proj = impl_base_1100 + cum_new_impl - cum_rec
    if impl_proj > impl_limit_1100:
        fig.add_annotation(
            x=wk, y=production_line[i],
            text="KPI Break",
            showarrow=True, arrowhead=2,
            ax=0, ay=-40,
            font=dict(color="red", size=12, weight="bold"),
            arrowcolor="red"
        )

# Líneas azules: Recovery Rate (real y proyectado)
fig.add_trace(go.Scatter(
    x=weeks_real, y=recovery_rate,
    mode="lines+markers+text",
    name="Recovery Rate",
    text=[f"{v}%" for v in recovery_rate],
    textposition="top center",
    line=dict(color="blue", width=2),
    yaxis="y2"
))
fig.add_trace(go.Scatter(
    x=weeks_prediction, y=recovery_rate_projection,
    mode="lines+markers+text",
    name="Recovery Rate (Projected)",
    text=[f"{v}%" for v in recovery_rate_projection],
    textposition="top center",
    line=dict(color="blue", width=2, dash="dot"),
    yaxis="y2", showlegend=False
))

fig.update_layout(
    title=dict(text="Backlog, Prediction Line and Recovery Rate Trend", font=dict(size=18)),
    xaxis=dict(title="Week", tickmode="linear", dtick=1, range=[32.5, 55.5]),
    yaxis=dict(title="Backlog / Production"),
    yaxis2=dict(title="Recovery Rate (%)", overlaying="y", side="right", range=[0, 100]),
    legend=dict(orientation="h", y=-0.2)
)

st.plotly_chart(fig, use_container_width=True, key="fig_recovery_overview")

# ==========================
# Annual Trend Charts
# ==========================
st.markdown("<h1>Annual CAPA Trends</h1>", unsafe_allow_html=True)

years = list(range(2020, TODAY.year + 1))
capa_unique = df_clean.groupby("capa number").agg({
    "created date": "min",
    "closed date": "max",
    "responsible site": "last"
}).reset_index()

created_per_year = capa_unique.groupby(capa_unique["created date"].dt.year)["capa number"].count()
closed_per_year = capa_unique.dropna(subset=["closed date"]).groupby(capa_unique["closed date"].dt.year)["capa number"].count()

inworks_counts = {}
for year in years:
    cutoff = pd.Timestamp(year=year, month=12, day=31)
    inworks_counts[year] = capa_unique[
        (capa_unique["created date"] <= cutoff) &
        ((capa_unique["closed date"].isna()) | (capa_unique["closed date"] > cutoff))
    ]["capa number"].nunique()

data_global = pd.DataFrame({
    "Year": years,
    "Created": created_per_year.reindex(years, fill_value=0).values,
    "Closed": closed_per_year.reindex(years, fill_value=0).values,
    "In works": pd.Series(inworks_counts).values
})

fig1 = go.Figure()
fig1.add_bar(x=data_global["Year"], y=data_global["Created"], name="Created", marker_color="skyblue",
             text=data_global["Created"], textposition="outside")
fig1.add_bar(x=data_global["Year"], y=data_global["Closed"], name="Closed", marker_color="orange",
             text=data_global["Closed"], textposition="outside")
fig1.add_trace(go.Scatter(x=data_global["Year"], y=data_global["In works"],
                          mode="lines+markers+text", name="In works",
                          text=data_global["In works"], textposition="top center", line=dict(color="red")))
fig1.update_layout(title=dict(text="Annual Global Trends", font=dict(size=18)), barmode="group")

# 1100 Annual
capa_1100 = capa_unique[capa_unique["responsible site"] == "1100"]
created_per_year_1100 = capa_1100.groupby(capa_1100["created date"].dt.year)["capa number"].count()
closed_per_year_1100 = capa_1100.dropna(subset=["closed date"]).groupby(capa_1100["closed date"].dt.year)["capa number"].count()

inworks_counts_1100 = {}
for year in years:
    cutoff = pd.Timestamp(year=year, month=12, day=31)
    inworks_counts_1100[year] = capa_1100[
        (capa_1100["created date"] <= cutoff) &
        ((capa_1100["closed date"].isna()) | (capa_1100["closed date"] > cutoff))
    ]["capa number"].nunique()

data_1100 = pd.DataFrame({
    "Year": years,
    "Created": created_per_year_1100.reindex(years, fill_value=0).values,
    "Closed": closed_per_year_1100.reindex(years, fill_value=0).values,
    "In works": pd.Series(inworks_counts_1100).values
})

fig2 = go.Figure()
fig2.add_bar(x=data_1100["Year"], y=data_1100["Created"], name="Created", marker_color="skyblue",
             text=data_1100["Created"], textposition="outside")
fig2.add_bar(x=data_1100["Year"], y=data_1100["Closed"], name="Closed", marker_color="orange",
             text=data_1100["Closed"], textposition="outside")
fig2.add_trace(go.Scatter(x=data_1100["Year"], y=data_1100["In works"],
                          mode="lines+markers+text", name="In works",
                          text=data_1100["In works"], textposition="top center", line=dict(color="red")))
fig2.update_layout(title=dict(text="Annual 1100 Trends", font=dict(size=18)), barmode="group")

col9, col10 = st.columns(2)
with col9:
    st.plotly_chart(fig1, use_container_width=True, key="fig_annual_global")
with col10:
    st.plotly_chart(fig2, use_container_width=True, key="fig_annual_1100")
