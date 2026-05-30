import io
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.stats import gaussian_kde
from PIL import Image

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ════════════════════════════════════════════════════

st.set_page_config(
    page_title="Dashboard Producción Palma",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "primary":  "#1b60a7",
    "success":  "#2ca02c",
    "danger":   "#d62728",
    "warning":  "#F1C40F",
    "info":     "#E74C3C",
    "bg":       "#F0F4F8",
}

RANKING_METRICS = ["TON/HA", "TON_ACEITE/HA", "EDAD"]
RANKING_SIZES   = [5, 10, 15, 20]

METRIC_DESCRIPTIONS = {
    "TON/HA":        "Toneladas de Racimos Frescos por Hectárea",
    "TON_ACEITE/HA": "Toneladas de Aceite por Hectárea (estimado)",
    "EDAD":          "Edad de la plantación (años)",
    "Frecuencia":    "Número de registros por categoría",
    "AREA":          "Área total cultivada (hectáreas)",
}

CATEGORICAL_NUMERIC_METRICS = ["TON/HA", "TON_ACEITE/HA", "EDAD"]

COLUMN_ALIASES = {
    "finca":    ["finca", "farm", "hacienda", "estacion"],
    "bloque":   ["bloque", "block", "blq"],
    "lote":     ["lote", "lot", "parcela", "plot"],
    "ano":      ["ano", "año", "year", "anio"],
    "siembra":  ["siembra", "año_siembra", "ano_siembra", "planting_year", "yr_siembra"],
    "edad":     ["edad", "age", "años_edad"],
    "material": ["material", "variedad", "variety", "genotipo", "genotype"],
    "area_ha":  ["area_ha", "area", "área", "area_hectareas", "hectareas", "ha", "superficie"],
    "n_palmas": ["n_palmas", "palmas", "num_palmas", "numero_palmas", "palms", "n palmas"],
    "densidad": ["densidad", "density", "dens"],
    "ton/lote": ["ton/lote", "ton_lote", "toneladas_lote", "produccion_lote", "prod_lote"],
    "ton/ha":   ["ton/ha", "ton_ha", "toneladas_ha", "produccion_ha", "prod_ha", "rff/ha"],
    "kg/palma": ["kg/palma", "kg_palma", "kilogramos_palma", "prod_palma"],
}

REQUIRED_INTERNAL = ["finca", "lote", "ano", "siembra", "material", "area_ha"]

# Helper para sanitizar keys
def sanitize_key(s):
    return str(s).replace(" ", "_").replace("/", "_").replace("-", "_").replace(".", "_").replace("(", "").replace(")", "")

# ════════════════════════════════════════════════════
# CSS PERSONALIZADO
# ════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #0A3D62 0%, #1A6B3C 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.8; font-size: 0.9rem; }

    /* KPI cards */
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #1b60a7;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .kpi-label { font-size: 0.72rem; font-weight: 600; color: #7A8899;
                 text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #1C2B3A; line-height: 1.1; }
    .kpi-sub   { font-size: 0.72rem; color: #7A8899; margin-top: 2px; }

    /* Sección separadora */
    .section-title {
        font-size: 1rem; font-weight: 700; color: #0A3D62;
        border-bottom: 2px solid #1A6B3C;
        padding-bottom: 0.3rem; margin: 1.2rem 0 0.8rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #F0F4F8; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600; font-size: 0.85rem;
    }

    /* Upload zone */
    .upload-zone {
        border: 2px dashed #1b60a7; border-radius: 10px;
        padding: 2rem; text-align: center; background: #f0f7ff;
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# 1. CARGA Y PREPARACIÓN DE DATOS
# ════════════════════════════════════════════════════

def _find_col(df_cols, aliases):
    for alias in aliases:
        if alias in df_cols:
            return alias
    return None

@st.cache_data(show_spinner=False)
def cargar_y_preparar_datos(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    df.columns = df.columns.str.strip().str.lower()

    cols_actual  = list(df.columns)
    rename_map   = {}
    missing_req  = []

    for internal, aliases in COLUMN_ALIASES.items():
        found = _find_col(cols_actual, aliases)
        if found:
            rename_map[found] = internal
        elif internal in REQUIRED_INTERNAL:
            missing_req.append(internal)

    if missing_req:
        raise ValueError(
            f"Faltan columnas obligatorias: {missing_req}\n"
            f"Columnas disponibles: {cols_actual}"
        )

    df = df.rename(columns=rename_map)
    df = df.rename(columns={
        "finca":    "FINCA",
        "bloque":   "BLOQUE",
        "lote":     "LOTE",
        "ano":      "ANO",
        "siembra":  "SIEMBRA",
        "edad":     "EDAD",
        "material": "MATERIAL",
        "area_ha":  "AREA",
        "n_palmas": "N_PALMAS",
        "densidad": "DENSIDAD",
        "ton/lote": "TON_LOTE",
        "ton/ha":   "TON/HA",
        "kg/palma": "KG/PALMA",
    })

    numeric_cols = ["ANO", "SIEMBRA", "EDAD", "AREA", "N_PALMAS",
                    "DENSIDAD", "TON_LOTE", "TON/HA", "KG/PALMA"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "TON/HA" not in df.columns or df["TON/HA"].isna().all():
        if "TON_LOTE" in df.columns and "AREA" in df.columns:
            df["TON/HA"] = np.where(df["AREA"] > 0, df["TON_LOTE"] / df["AREA"], np.nan)

    if "KG/PALMA" not in df.columns or df["KG/PALMA"].isna().all():
        if "TON_LOTE" in df.columns and "N_PALMAS" in df.columns:
            df["KG/PALMA"] = np.where(
                df["N_PALMAS"] > 0,
                df["TON_LOTE"] * 1000.0 / df["N_PALMAS"],
                np.nan
            )

    df["EDAD"] = df["ANO"] - df["SIEMBRA"]
    df["TON_ACEITE/HA"] = df["TON/HA"] * float(0.2)

    data = df.copy()
    data = data[data["ANO"].notna()]
    data = data[data["SIEMBRA"].notna()]
    data = data[data["TON/HA"].notna()]
    data = data[data["EDAD"] >= 0]

    data["LOTE"]    = data["LOTE"].astype(str)
    data["NOM_HAC"] = data["LOTE"].str.extract(r"^([A-Za-z]+)")[0].fillna("SIN_GRUPO")
    data["CONCAT"]  = data["NOM_HAC"].astype(str) + " - " + data["LOTE"].astype(str)

    return data


# ════════════════════════════════════════════════════
# 2. FUNCIONES DE GRÁFICOS
# ════════════════════════════════════════════════════

def crear_treemap(X: pd.DataFrame, indicator: str) -> go.Figure:
    if X.empty or indicator not in X.columns:
        return go.Figure()

    fig = px.treemap(
        X,
        path=[px.Constant(" "), "NOM_HAC", "CONCAT"],
        hover_data=[indicator],
        custom_data=["EDAD", "MATERIAL", "AREA", "TON/HA", "TON_ACEITE/HA"],
        values=X[indicator].abs() + 0.01,
        color=X[indicator],
        color_continuous_scale="Spectral",
        color_continuous_midpoint=np.average(
            X[indicator].replace([np.inf, -np.inf], np.nan).dropna()
        ),
    )
    fig.data[0].textinfo     = "label+text+value"
    fig.data[0].texttemplate = (
        "%{label}<br><br>EDAD: %{customdata[0]:.1f} años<br>"
        "MATERIAL: %{customdata[1]}<br>"
        "AREA: %{customdata[2]:.1f} ha<br>"
        "TON/HA: %{customdata[3]:.2f}<br>"
        "TON_ACEITE/HA: %{customdata[4]:.2f}"
    )
    fig.update_traces(hovertemplate="%{label}<br>%{value:.2f}")
    fig.update_layout(margin=dict(t=10, l=0, r=0, b=0), height=420)
    return fig


def crear_ranking_charts(df, metric, top_n, group_col, group_label):
    X = df.groupby(group_col)[metric].mean().reset_index()
    X[group_col] = X[group_col].astype(str).str.strip()
    X = X.sort_values(by=metric, ascending=False)
    X.columns = ["Group", "Value"]

    n = len(X)
    if n == 0:
        return go.Figure(), go.Figure()

    actual_n = min(top_n, n)

    # Bajos
    chart_low = px.bar(
        X, x="Value", y="Group",
        range_y=[n - actual_n - 0.5, n],
        color_discrete_sequence=[COLORS["danger"]],
        title=f"🔴 Top {actual_n} {group_label}s con menor {metric}",
    )
    chart_low.update_layout(
        height=280, margin=dict(t=40, l=0, r=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, title_font_size=12,
    )
    chart_low.update_xaxes(title_text=f"Promedio {metric}")
    chart_low.update_yaxes(title_text=group_label)

    # Altos
    X_asc = X.sort_values(by="Value", ascending=True)
    chart_high = px.bar(
        X_asc, x="Value", y="Group",
        range_y=[n - actual_n - 0.5, n],
        color_discrete_sequence=[COLORS["success"]],
        title=f"🟢 Top {actual_n} {group_label}s con mayor {metric}",
    )
    chart_high.update_layout(
        height=280, margin=dict(t=40, l=0, r=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, title_font_size=12,
    )
    chart_high.update_xaxes(title_text=f"Promedio {metric}")
    chart_high.update_yaxes(title_text=group_label)

    return chart_high, chart_low


def crear_distplot_manual(vals: pd.Series, metric: str) -> go.Figure:
    """Histograma + KDE sin depender de ff.create_distplot (deprecado)."""
    if len(vals) < 2:
        return go.Figure()

    kde_x = np.linspace(vals.min(), vals.max(), 200)
    kde_y = gaussian_kde(vals)(kde_x)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=vals, histnorm="probability density",
        name="Histograma", marker_color=COLORS["primary"],
        opacity=0.6, nbinsx=20,
    ))
    fig.add_trace(go.Scatter(
        x=kde_x, y=kde_y, mode="lines",
        name="KDE", line=dict(color=COLORS["primary"], width=2),
    ))

    mean_val   = float(vals.mean())
    median_val = float(vals.median())

    fig.add_vline(x=mean_val,   line_dash="dash", line_color=COLORS["info"],
                  annotation_text=f"Media: {mean_val:.2f}",
                  annotation_position="top right")
    fig.add_vline(x=median_val, line_dash="dot",  line_color=COLORS["warning"],
                  annotation_text=f"Mediana: {median_val:.2f}",
                  annotation_position="bottom right")

    fig.update_layout(
        height=260, margin=dict(t=10, l=0, r=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, xaxis_title=metric, yaxis_title="Densidad",
    )
    return fig


def crear_boxplot(vals: pd.Series, metric: str) -> go.Figure:
    fig = go.Figure(go.Box(
        y=vals.values.tolist(), name=metric,
        jitter=0.15, marker=dict(color=COLORS["primary"]),
    ))
    fig.update_layout(
        height=260, margin=dict(t=10, l=0, r=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, yaxis=dict(zeroline=False),
    )
    return fig


def crear_tabla_estadisticas(vals: pd.Series) -> go.Figure:
    desc = vals.describe().astype(float)
    cv   = (desc["std"] / desc["mean"] * 100) if desc["mean"] != 0 else 0.0
    skew = float(vals.skew()) if len(vals) > 2 else np.nan
    kurt = float(vals.kurt()) if len(vals) > 3 else np.nan

    stats_df = pd.DataFrame({
        "Parámetro": ["N", "Media", "Mediana", "Desv. Estándar",
                      "Mínimo", "Máximo", "CV (%)", "Asimetría", "Kurtosis"],
        "Valor": [desc["count"], desc["mean"], float(vals.median()),
                  desc["std"], desc["min"], desc["max"], cv, skew, kurt],
    }).round(3)

    fig = go.Figure(go.Table(
        header=dict(values=["Parámetro", "Valor"],
                    fill_color=COLORS["primary"],
                    font=dict(color="white", size=12), align="left"),
        cells=dict(values=[stats_df["Parámetro"], stats_df["Valor"]],
                   fill_color="white", align="left"),
    ))
    fig.update_layout(
        height=300, margin=dict(t=0, l=0, r=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def crear_indicador_comparativo(X, X_ref, col, label, scope, year):
    fig = go.Figure(go.Indicator(
        mode="number+delta",
        value=X[col].dropna().mean(),
        title={"text": f"{label}<br><span style='font-size:0.8em;color:gray'>Ref: {scope} | Año: {year}</span>",
               "font": {"size": 16}},
        delta={"position": "top",
               "reference": X_ref[col].dropna().mean(),
               "relative": True, "valueformat": ".2%"},
        number={"font": {"size": 22}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=160, margin=dict(t=10, l=0, r=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def crear_tendencia_anual(data_num, metric):
    Y = data_num.groupby("ANO")[metric].agg(["mean", "std"]).reset_index().dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=Y["ANO"], y=Y["mean"], mode="lines+markers",
        name="Media", line=dict(color=COLORS["primary"], width=2),
        hovertemplate=f"Año: %{{x}}<br>Media {metric}: %{{y:.2f}}",
    ))
    fig.add_trace(go.Scatter(
        x=Y["ANO"], y=Y["std"], mode="lines+markers",
        name="Desv. Estándar", line=dict(color=COLORS["warning"], width=2, dash="dot"),
        hovertemplate=f"Año: %{{x}}<br>DE {metric}: %{{y:.2f}}",
    ))
    fig.update_layout(
        height=260, margin=dict(t=10, l=0, r=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.1),
        xaxis_title="Año", yaxis_title=metric,
    )
    return fig


def crear_heatmap_correlacion(corr_matrix, metric):
    matrix = corr_matrix[[metric]].drop(
        [c for c in [metric, "ANO"] if c in corr_matrix.index], axis=0, errors="ignore"
    )
    if matrix.empty:
        return go.Figure()
    fig = px.imshow(
        matrix, color_continuous_scale="RdBu_r",
        aspect="auto", text_auto=True,
        labels={"color": f"Corr({metric}, ·)"},
        zmin=-1, zmax=1,
    )
    fig.update_layout(
        height=260, margin=dict(t=10, l=0, r=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_bar_table_pie(df_agg, cat_col, value_label):
    X = df_agg.copy().sort_values("Count", ascending=False).round(2)
    n = len(X)
    if n == 0:
        return go.Figure(), go.Figure(), go.Figure()

    bar = px.bar(X, x="Count", y=cat_col,
                 color_discrete_sequence=[COLORS["primary"]])
    bar.update_layout(height=220, margin=dict(t=0, l=0, r=0, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      showlegend=False, xaxis_title=value_label, yaxis_title="")

    table = go.Figure(go.Table(
        header=dict(values=[cat_col, value_label],
                    fill_color=COLORS["primary"],
                    font=dict(color="white", size=11), align="left"),
        cells=dict(values=[X[cat_col], X["Count"]],
                   fill_color="white", align="left"),
    ))
    table.update_layout(height=220, margin=dict(t=0, l=0, r=0, b=0),
                        paper_bgcolor="rgba(0,0,0,0)")

    top5 = X.head(5).copy()
    top5["Pct"] = (top5["Count"] / top5["Count"].sum() * 100).round(2)
    pie = go.Figure(go.Pie(labels=top5[cat_col], values=top5["Pct"]))
    pie.update_traces(textposition="inside", textinfo="label+percent")
    pie.update_layout(height=220, margin=dict(t=0, l=0, r=0, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", showlegend=False)

    return bar, table, pie


# ════════════════════════════════════════════════════
# 3. SECCIONES DEL DASHBOARD
# ════════════════════════════════════════════════════

def seccion_kpis(data: pd.DataFrame):
    years = sorted(data["ANO"].dropna().unique())
    year_max = int(years[-1]) if years else "—"

    cols = st.columns(6)
    kpis = [
        ("Año de análisis",       str(year_max),                    "📅"),
        ("Fincas",                str(data["FINCA"].nunique()),      "🏡"),
        ("Lotes analizados",      str(data["LOTE"].nunique()),       "🌿"),
        ("Registros",             f"{len(data):,}",                  "📋"),
        ("TON/HA promedio",       f"{data['TON/HA'].mean():.2f}",    "📊"),
        ("TON_ACEITE/HA prom.",   f"{data['TON_ACEITE/HA'].mean():.2f}", "🛢️"),
    ]
    for col, (label, value, icon) in zip(cols, kpis):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{icon} {label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)


def seccion_treemaps_rankings(data, data_by_year, years_desc, year_sel):
    X = data_by_year[year_sel].fillna(0)

    # ── Treemaps ──
    st.markdown('<div class="section-title">🗺️ Mapas de árbol (Treemaps)</div>',
                unsafe_allow_html=True)
    ind_sel = st.radio("Indicador treemap:", ["TON/HA", "TON_ACEITE/HA"],
                       horizontal=True, key="treemap_ind")
    treemap_key = sanitize_key(f"treemap_{ind_sel}_{int(year_sel)}")
    st.plotly_chart(crear_treemap(X, ind_sel), use_container_width=True, key=treemap_key)

    # ── Rankings ──
    st.markdown('<div class="section-title">🏆 Rankings de producción</div>',
                unsafe_allow_html=True)

    metric_sel = st.selectbox("Métrica de ranking:", RANKING_METRICS, key="rank_metric")
    top_n_sel  = st.select_slider("Tamaño del ranking:", RANKING_SIZES, value=10, key="rank_n")
    group_sel  = st.radio("Agrupar por:", ["Grupos de Lotes (NOM_HAC)", "Lotes individuales"],
                          horizontal=True, key="rank_group")

    group_col   = "NOM_HAC" if "Grupos" in group_sel else "LOTE"
    group_label = "Grupo" if "Grupos" in group_sel else "Lote"

    ch, cl = crear_ranking_charts(X, metric_sel, top_n_sel, group_col, group_label)
    c1, c2 = st.columns(2)
    key_high = sanitize_key(f"ranking_high_{metric_sel}_{top_n_sel}_{group_col}_{int(year_sel)}")
    key_low  = sanitize_key(f"ranking_low_{metric_sel}_{top_n_sel}_{group_col}_{int(year_sel)}")
    c1.plotly_chart(ch, use_container_width=True, key=key_high)
    c2.plotly_chart(cl, use_container_width=True, key=key_low)

    # ── Indicadores comparativos ──
    st.markdown('<div class="section-title">📈 Indicadores comparativos</div>',
                unsafe_allow_html=True)

    scope_options = ["global"] + [str(int(y)) for y in years_desc]
    scope_sel     = st.selectbox("Comparar contra:", scope_options, key="scope_sel")
    X_ref = data if scope_sel == "global" else data_by_year.get(float(scope_sel), data)

    ind_cols = [c for c in ["TON/HA", "TON_ACEITE/HA", "EDAD", "AREA"] if c in X.columns]
    cols_ind = st.columns(len(ind_cols)) if ind_cols else []
    for col_ui, col_name in zip(cols_ind, ind_cols):
        fig = crear_indicador_comparativo(X, X_ref, col_name, col_name, scope_sel, year_sel)
        indicator_key = sanitize_key(f"indicator_{col_name}_{scope_sel}_{int(year_sel)}")
        col_ui.plotly_chart(fig, use_container_width=True, key=indicator_key)


def seccion_variables_numericas(data, data_by_year, years_desc):
    data_num = data.select_dtypes(include=["number"]).copy()
    corr_mat = data_num.corr(method="pearson").round(3)

    num_cols = [c for c in data_num.columns if c != "ANO"]
    if not num_cols:
        st.info("No hay variables numéricas disponibles.")
        return

    metric_sel = st.selectbox("Variable numérica:", num_cols, key="num_metric")

    scope_options = ["global"] + [str(int(y)) for y in years_desc]
    scope_sel     = st.selectbox("Periodo:", scope_options, key="num_scope")

    if scope_sel == "global":
        X_num = data_num
    else:
        X_num = data_num[data_num["ANO"] == float(scope_sel)]

    vals = X_num[metric_sel].replace([np.inf, -np.inf], np.nan).dropna()

    if len(vals) == 0:
        st.warning("Sin datos para esta selección.")
        return

    st.markdown(f'<div class="section-title">📊 Distribución de {metric_sel} — {scope_sel}</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    key_dist = sanitize_key(f"dist_{metric_sel}_{scope_sel}")
    key_box  = sanitize_key(f"box_{metric_sel}_{scope_sel}")
    key_stats = sanitize_key(f"stats_{metric_sel}_{scope_sel}")
    c1.plotly_chart(crear_distplot_manual(vals, metric_sel), use_container_width=True, key=key_dist)
    c2.plotly_chart(crear_boxplot(vals, metric_sel),         use_container_width=True, key=key_box)
    c3.plotly_chart(crear_tabla_estadisticas(vals),          use_container_width=True, key=key_stats)

    st.markdown('<div class="section-title">🔗 Correlaciones y tendencia temporal</div>',
                unsafe_allow_html=True)

    c4, c5 = st.columns(2)
    key_heat = sanitize_key(f"heat_{metric_sel}_{scope_sel}")
    key_trend = sanitize_key(f"trend_{metric_sel}_{scope_sel}")
    c4.plotly_chart(crear_heatmap_correlacion(corr_mat, metric_sel), use_container_width=True, key=key_heat)
    c5.plotly_chart(crear_tendencia_anual(data_num, metric_sel),     use_container_width=True, key=key_trend)


def seccion_variables_categoricas(data, data_by_year, years_desc):
    data_cat = data.select_dtypes(include=["object"]).copy()
    cat_cols  = [c for c in data_cat.columns
                 if c not in ("NOM_HAC", "CONCAT") and data_cat[c].nunique() < 50]

    if not cat_cols:
        st.info("No hay variables categóricas disponibles.")
        return

    cat_sel   = st.selectbox("Variable categórica:", cat_cols, key="cat_metric")
    scope_options = ["global"] + [str(int(y)) for y in years_desc]
    scope_sel = st.selectbox("Periodo:", scope_options, key="cat_scope")

    base = data if scope_sel == "global" else data_by_year.get(float(scope_sel), data)

    metric_tabs = ["Frecuencia", "Área (ha)"] + CATEGORICAL_NUMERIC_METRICS
    tab_objs    = st.tabs(metric_tabs)

    # Frecuencia
    with tab_objs[0]:
        cnt_df = (base.groupby(cat_sel)
                      .size().reset_index(name="Count"))
        bar, tbl, pie = make_bar_table_pie(cnt_df, cat_sel, "Frecuencia")
        c1, c2, c3 = st.columns(3)
        key_bar = sanitize_key(f"cat_bar_{cat_sel}_{scope_sel}_freq")
        key_tbl = sanitize_key(f"cat_table_{cat_sel}_{scope_sel}_freq")
        key_pie = sanitize_key(f"cat_pie_{cat_sel}_{scope_sel}_freq")
        c1.plotly_chart(bar, use_container_width=True, key=key_bar)
        c2.plotly_chart(tbl, use_container_width=True, key=key_tbl)
        c3.plotly_chart(pie, use_container_width=True, key=key_pie)

    # Área
    with tab_objs[1]:
        if "AREA" in base.columns:
            area_df = (base.groupby(cat_sel)["AREA"]
                           .sum().reset_index().rename(columns={"AREA": "Count"}))
            bar, tbl, pie = make_bar_table_pie(area_df, cat_sel, "Área (ha)")
            c1, c2, c3 = st.columns(3)
            key_bar_a = sanitize_key(f"cat_bar_{cat_sel}_{scope_sel}_area")
            key_tbl_a = sanitize_key(f"cat_table_{cat_sel}_{scope_sel}_area")
            key_pie_a = sanitize_key(f"cat_pie_{cat_sel}_{scope_sel}_area")
            c1.plotly_chart(bar, use_container_width=True, key=key_bar_a)
            c2.plotly_chart(tbl, use_container_width=True, key=key_tbl_a)
            c3.plotly_chart(pie, use_container_width=True, key=key_pie_a)
        else:
            st.info("Columna AREA no disponible.")

    # Métricas numéricas
    for i, num_metric in enumerate(CATEGORICAL_NUMERIC_METRICS):
        with tab_objs[i + 2]:
            if num_metric in base.columns:
                m_df = (base.groupby(cat_sel)[num_metric]
                            .mean().reset_index().rename(columns={num_metric: "Count"}))
                bar, tbl, pie = make_bar_table_pie(m_df, cat_sel, num_metric)
                key_bar_n = sanitize_key(f"catnum_bar_{num_metric}_{cat_sel}_{scope_sel}_{i}")
                key_tbl_n = sanitize_key(f"catnum_table_{num_metric}_{cat_sel}_{scope_sel}_{i}")
                key_pie_n = sanitize_key(f"catnum_pie_{num_metric}_{cat_sel}_{scope_sel}_{i}")
                c1, c2, c3 = st.columns(3)
                c1.plotly_chart(bar, use_container_width=True, key=key_bar_n)
                c2.plotly_chart(tbl, use_container_width=True, key=key_tbl_n)
                c3.plotly_chart(pie, use_container_width=True, key=key_pie_n)
            else:
                st.info(f"Columna {num_metric} no disponible.")


def seccion_tabla_datos(data, data_by_year, years_desc, year_sel):
    st.markdown('<div class="section-title">📋 Tabla de datos detallada</div>',
                unsafe_allow_html=True)

    show_all = st.checkbox("Mostrar todos los años", value=False)
    df_show  = data if show_all else data_by_year[year_sel]

    # Filtros rápidos
    c1, c2 = st.columns(2)
    fincas  = ["Todas"] + sorted(df_show["FINCA"].dropna().unique().tolist())
    mats    = ["Todos"] + sorted(df_show["MATERIAL"].dropna().unique().tolist())
    finca_f = c1.selectbox("Filtrar por Finca:", fincas, key="tbl_finca")
    mat_f   = c2.selectbox("Filtrar por Material:", mats, key="tbl_mat")

    if finca_f != "Todas":
        df_show = df_show[df_show["FINCA"] == finca_f]
    if mat_f != "Todos":
        df_show = df_show[df_show["MATERIAL"] == mat_f]

    st.dataframe(
        df_show.sort_values(["ANO", "FINCA", "LOTE"]).reset_index(drop=True),
        use_container_width=True, height=420,
    )

    # Descarga
    csv = df_show.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar CSV filtrado", csv,
        file_name=f"produccion_filtrado_{year_sel}.csv",
        mime="text/csv",
    )


# ════════════════════════════════════════════════════
# 4. SIDEBAR
# ════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        try:
            img = Image.open("logo_sidebar.png")
            st.image(img, width=260)
        except Exception:
            st.markdown("## 🌴 Dashboard Producción")

        st.markdown("---")

        st.markdown("### 📂 Cargar datos")
        uploaded = st.file_uploader(
            "Sube tu archivo Excel (.xlsx)",
            type=["xlsx", "xls"],
            help="El archivo debe contener columnas de finca, lote, año, siembra, material, área y ton/ha.",
        )

        st.markdown("---")
        st.markdown("### ℹ️ Columnas requeridas")
        st.markdown("""
        | Interno | Variantes aceptadas |
        |---------|---------------------|
        | finca | finca, farm, hacienda |
        | lote | lote, lot, parcela |
        | ano | ano, año, year |
        | siembra | siembra, ano_siembra |
        | material | material, variedad |
        | area_ha | area_ha, area, ha |
        | ton/ha | ton/ha, ton_ha, rff/ha |
        """)

    return uploaded


# ════════════════════════════════════════════════════
# 5. MAIN
# ════════════════════════════════════════════════════

def main():
    uploaded = render_sidebar()

    # ── Header ──
    st.markdown(f"""
    <div class="main-header">
        <h1>🌴 Dashboard de Producción - Farmprecision</h1>
        <p>Análisis integral de productividad de palma de aceite</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sin archivo ──
    if uploaded is None:
        st.markdown("""
        <div class="upload-zone">
            <h3>📂 Sube tu archivo Excel para comenzar</h3>
            <p>Usa el panel lateral para cargar el archivo de producción histórica.</p>
            <p><strong>Formatos soportados:</strong> .xlsx · .xls</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📌 ¿Qué puedes analizar?")
        c1, c2, c3 = st.columns(3)
        c1.info("**🗺️ Treemaps** jerárquicos por grupo de lote y lote individual")
        c2.info("**🏆 Rankings** de los mejores y peores lotes por TON/HA, TON_ACEITE/HA y Edad")
        c3.info("**📊 Estadísticas** descriptivas, distribuciones, correlaciones y tendencias")
        return

    # ── Carga de datos ──
    with st.spinner("⏳ Procesando datos..."):
        try:
            data = cargar_y_preparar_datos(uploaded.read())
        except ValueError as e:
            st.error(f"❌ Error al cargar el archivo:\n\n{e}")
            return
        except Exception as e:
            st.error(f"❌ Error inesperado: {e}")
            return

    if data.empty:
        st.warning("⚠️ El archivo no contiene datos válidos después del filtrado.")
        return

    # ── Preparar datos por año ──
    years      = np.sort(data["ANO"].dropna().unique())
    years_desc = years[::-1]
    data_by_year = {y: data[data["ANO"] == y].copy() for y in years}

    # ── KPIs ──
    seccion_kpis(data)

    st.markdown("---")

    # ── Selector de año (global para Tab 1 y Tab 4) ──
    year_sel = st.selectbox(
        "📅 Año de análisis principal:",
        options=[int(y) for y in years_desc],
        key="year_global",
    )

    # ── Tabs principales ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Treemaps & Rankings",
        "📊 Variables Numéricas",
        "🏷️ Variables Categóricas",
        "📋 Tabla de Datos",
    ])

    with tab1:
        seccion_treemaps_rankings(data, data_by_year, years_desc, float(year_sel))

    with tab2:
        seccion_variables_numericas(data, data_by_year, years_desc)

    with tab3:
        seccion_variables_categoricas(data, data_by_year, years_desc)

    with tab4:
        seccion_tabla_datos(data, data_by_year, years_desc, float(year_sel))


if __name__ == "__main__":
    main()