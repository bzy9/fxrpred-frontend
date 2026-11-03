"""Data Resources view. All results are local benchmark summaries."""

from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_resources import (
    DATASETS,
    DESCRIPTORS,
    SOURCE_COLUMN,
    BenchmarkDataError,
    benchmark_csv_bytes,
    dataset_readme,
    descriptor_profiles,
    filter_benchmark,
    label_counts,
    load_benchmark_data,
    load_descriptor_cache,
    preview_frame,
    task_summary,
)


COLORS = {"Positive (1)": "#3b82f6", "Negative (0)": "#94a3b8", "Missing label": "#d8b4fe"}
PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
CSS = """
<style>
[data-testid="stAppViewContainer"]:has(.st-key-resources_page) { background: #f8fafc !important; }
.st-key-resources_page { padding-bottom: 50px; }
.resource-hero {
    background: linear-gradient(135deg, #E0E7FF 0%, #F8FAFC 50%, #E0F2FE 100%);
    border-radius: 16px; padding: 30px;
    box-shadow: 0 20px 40px rgba(79,70,229,.05), 0 1px 3px rgba(0,0,0,.03);
    margin-top: 20px; margin-bottom: 40px;
    border: 1px solid rgba(79,70,229,.1);
}
.resource-title { font-size: 38px; font-weight: 900; color: #1e293b; margin-bottom: 15px; }
.resource-intro { color: #475569; font-size: 16px; line-height: 1.8; }
.resource-tag {
    display: inline-block; background-color: white; padding: 6px 18px; border-radius: 20px;
    font-size: 14px; font-weight: 700; color: #3b82f6;
    box-shadow: 0 4px 6px rgba(0,0,0,.02); margin-right: 12px; margin-top: 20px;
    border: 1px solid #bfdbfe;
}
.resource-metric {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 13px;
    padding: 19px 17px; margin: 0 0 8px; min-height: 107px;
    box-shadow: 0 4px 12px rgba(15,23,42,.025);
}
.resource-metric-value { font-size: 28px; font-weight: 850; color: #2563eb; line-height: 1.2; }
.resource-metric-label { color: #64748b; font-size: 12px; margin-top: 10px; line-height: 1.5; }
.resource-section {
    font-size: 15px; font-weight: 700; display: block;
    padding: 8px 16px; border-radius: 8px;
    margin-top: 10px; margin-bottom: 15px;
}
.resource-section-query { background-color: #E0F2FE; color: #075985; border-left: 5px solid #0284C7; }
.resource-section-overview { background-color: #DCFCE7; color: #166534; border-left: 5px solid #16A34A; }
.resource-section-downloads { background-color: #E6FFFA; color: #0D9488; border-left: 5px solid #0D9488; }
.resource-chart-title { font-size: 14px; font-weight: 750; color: #334155; margin: 12px 0 3px; }
.resource-results { color: #475569; font-size: 13px; line-height: 1.7; margin: 17px 0 13px; }
.st-key-resources_page [data-testid="stVerticalBlock"] { gap: .35rem !important; }
.st-key-resources_page [data-baseweb="tab-list"] { gap: 12px; border-bottom: 1px solid #e2e8f0; }
.st-key-resources_page [data-baseweb="tab"] { font-weight: 700; padding-left: 13px; padding-right: 13px; }
.st-key-resources_page [data-testid="stDownloadButton"] button {
    background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe;
    border-radius: 9px; min-height: 44px; margin-top: 8px; font-weight: 650;
}
.st-key-resources_page [data-testid="stDownloadButton"] button:hover { background: #dbeafe; border-color: #93c5fd; }
.st-key-resources_page [data-testid="stButton"] button { border-radius: 8px; }
.st-key-resources_page [data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    box-shadow: 0 4px 10px rgba(79,70,229,.3) !important;
    font-size: 16px !important; font-weight: bold !important; padding: 10px 0 !important;
}
.st-key-resources_page [data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-2px); box-shadow: 0 6px 15px rgba(79,70,229,.4) !important;
}
@media (max-width: 768px) {
    .resource-metric { padding: 15px 12px; }
    .resource-metric-value { font-size: 24px; }
}
</style>
"""


@st.cache_data(show_spinner=False)
def _cached_benchmark(data_dir: str, dataset_key: str, file_signature: tuple):
    return load_benchmark_data(Path(data_dir), dataset_key)


@st.cache_data(show_spinner=False)
def _cached_descriptors(data_dir: str, file_signature: tuple):
    return load_descriptor_cache(Path(data_dir))


def _signature(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _section(title: str) -> None:
    variant = {"Data query": "query", "Dataset Overview": "overview", "Downloads": "downloads"}[title]
    st.markdown(
        f'<div class="resource-section resource-section-{variant}" role="heading" aria-level="2">{escape(title)}</div>',
        unsafe_allow_html=True,
    )


def _metric(value: str, label: str) -> None:
    st.markdown(
        '<div class="resource-metric">'
        f'<div class="resource-metric-value">{escape(value)}</div>'
        f'<div class="resource-metric-label">{escape(label)}</div></div>',
        unsafe_allow_html=True,
    )


def _plot_style(figure: go.Figure, height: int = 300) -> go.Figure:
    figure.update_layout(
        template="plotly_white", height=height,
        margin=dict(l=8, r=12, t=16, b=15),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", size=12, color="#475569"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        hoverlabel=dict(bgcolor="white"),
    )
    figure.update_xaxes(gridcolor="#edf2f7", zeroline=False)
    figure.update_yaxes(gridcolor="#edf2f7", zeroline=False)
    return figure


def _overview(frame: pd.DataFrame, dataset_key: str, cache) -> None:
    _section("Dataset Overview")
    spec = DATASETS[dataset_key]
    summary = task_summary(frame, dataset_key)
    columns = st.columns(4, gap="medium")
    if dataset_key == "MTL":
        labeled = int(summary["Labeled"].sum())
        coverage = labeled / (len(frame) * len(spec["labels"]))
        metrics = [
            (f"{len(frame):,}", "Compound records"),
            (str(len(spec["labels"])), "Prediction tasks"),
            (f"{labeled:,}", "Annotated task entries"),
            (f"{coverage:.1%}", "Task-label coverage"),
        ]
    else:
        counts = label_counts(frame, "label")
        metrics = [
            (f"{len(frame):,}", "Compound records"),
            (f"{counts['positive']:,}", f"{spec['positive']} records · label 1"),
            (f"{counts['negative']:,}", f"{spec['negative']} records · label 0"),
            (f"{counts['positive'] / len(frame):.1%}", "Positive-label fraction"),
        ]
    for column, (value, label) in zip(columns, metrics):
        with column:
            _metric(value, label)

    if dataset_key == "MTL":
        label_plot, split_plot = st.columns([1.4, 1], gap="large")
        with label_plot:
            st.markdown('<div class="resource-chart-title">Task label distributions</div>', unsafe_allow_html=True)
            figure = go.Figure()
            for name, color in [("Positive", COLORS["Positive (1)"]), ("Negative", COLORS["Negative (0)"]), ("Missing", COLORS["Missing label"])]:
                figure.add_bar(
                    name=name, y=summary["Task"], x=summary[name], orientation="h",
                    marker_color=color, hovertemplate=f"%{{y}} · {name}<br>%{{x:,}} records<extra></extra>",
                )
            figure.update_layout(barmode="stack")
            figure.update_xaxes(title_text="Compound records")
            figure.update_yaxes(autorange="reversed")
            st.plotly_chart(_plot_style(figure, 290), use_container_width=True, config=PLOT_CONFIG, key="resources_mtl_labels")
        with split_plot:
            groups = frame["group"].value_counts()
            order = [g for g in ("training", "val", "test") if g in groups.index]
            order += [g for g in groups.index if g not in order]
            split_chart = go.Figure(go.Bar(
                x=[{"training": "Training", "val": "Validation", "test": "Test"}.get(g, g) for g in order],
                y=[int(groups[g]) for g in order], marker_color=["#60a5fa", "#a78bfa", "#5eead4"][:len(order)],
                text=[f"{int(groups[g]):,}" for g in order], textposition="auto",
                hovertemplate="%{x}<br>%{y:,} records<extra></extra>",
            ))
            split_chart.update_yaxes(title_text="Compound records")
            st.markdown('<div class="resource-chart-title">Original dataset partitions</div>', unsafe_allow_html=True)
            st.plotly_chart(_plot_style(split_chart, 290), use_container_width=True, config=PLOT_CONFIG, key="resources_mtl_splits")
        visible = summary.copy()
        visible["Coverage (%)"] = (visible.pop("Coverage") * 100).round(1)
        st.dataframe(visible, hide_index=True, use_container_width=True)
    else:
        counts = label_counts(frame, "label")
        figure = go.Figure(go.Bar(
            y=[f"{spec['positive']} (1)", f"{spec['negative']} (0)"],
            x=[counts["positive"], counts["negative"]], orientation="h",
            marker_color=[COLORS["Positive (1)"], COLORS["Negative (0)"]],
            text=[f"{counts['positive']:,}", f"{counts['negative']:,}"],
            textposition="auto", hovertemplate="%{y}<br>%{x:,} records<extra></extra>",
        ))
        figure.update_xaxes(title_text="Compound records")
        figure.update_yaxes(autorange="reversed")
        st.plotly_chart(_plot_style(figure, 230), use_container_width=True, config=PLOT_CONFIG, key=f"resources_{dataset_key}_labels")

    _profiles(frame, cache, dataset_key)


def _profiles(frame: pd.DataFrame, cache, dataset_key: str) -> None:
    if cache is None:
        st.info("Molecular-property profiles are unavailable in this copy.")
        return
    label_column = "FXR_ago" if dataset_key == "MTL" else "label"
    profiles = descriptor_profiles(frame, cache, label_column)
    columns = st.columns(3, gap="medium")
    for index, (name, (title, axis_title)) in enumerate(DESCRIPTORS.items()):
        with columns[index]:
            st.markdown(f'<div class="resource-chart-title">{escape(title)}</div>', unsafe_allow_html=True)
            fig = go.Figure(go.Histogram(
                x=profiles[name].dropna(), marker_color="#60a5fa",
                histnorm="percent", nbinsx=40,
                hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
            ))
            fig.update_layout(showlegend=False)
            fig.update_xaxes(title_text=axis_title)
            fig.update_yaxes(title_text="Frequency (%)")
            st.plotly_chart(_plot_style(fig, 245), use_container_width=True, config=PLOT_CONFIG, key=f"resources_{dataset_key}_{name}")


def _browse(frame: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    _section("Data query")
    prefix = f"resources_{dataset_key}"
    with st.form(key=f"{prefix}_search_form", clear_on_submit=False, border=False):
        query_col, field_col, button_col = st.columns([2.4, 1, .7], gap="medium", vertical_alignment="bottom")
        with query_col:
            query = st.text_input(
                "Search benchmark", placeholder="SMILES fragment, NURA_00002 or CHEMBL ID",
                help="SMILES text is case-sensitive. Source identifiers ignore case. This is literal text search, not substructure search.",
                key=f"{prefix}_query",
            )
        with field_col:
            search_field = st.selectbox("Search in", ["SMILES or source ID", "SMILES", "Source ID"], key=f"{prefix}_search_field")
        with button_col:
            submitted = st.form_submit_button("Search", type="primary", use_container_width=True)
    criteria_key = f"{prefix}_submitted_search"
    if submitted:
        st.session_state[criteria_key] = {"query": query, "search_field": search_field}
    criteria = st.session_state.get(criteria_key, {"query": "", "search_field": "SMILES or source ID"})
    label_column = "FXR_ago" if dataset_key == "MTL" else "label"
    filtered = filter_benchmark(frame, criteria["query"], criteria["search_field"], label_column=label_column)
    limit = 20
    st.markdown(
        f'<div class="resource-results"><b>{len(filtered):,}</b> matching records out of {len(frame):,} '
        f'· showing the first {min(limit, len(filtered)):,}</div>', unsafe_allow_html=True,
    )
    if filtered.empty:
        st.info("No matching records. Shorten or clear the search text.")
    else:
        st.dataframe(
            preview_frame(filtered, dataset_key, limit), hide_index=True, use_container_width=True,
            height=min(560, 45 + 35 * min(limit, len(filtered))),
            column_config={
                "smiles": st.column_config.TextColumn("SMILES", width="large"),
                SOURCE_COLUMN: st.column_config.TextColumn("Source ID(s)", width="large", help="Available provenance identifiers; not included in CSV downloads."),
            },
        )
    return filtered


def _downloads(frame: pd.DataFrame, dataset_key: str, filtered: pd.DataFrame) -> None:
    _section("Downloads")
    prefix = f"resources_{dataset_key}"
    if dataset_key == "MTL":
        st.download_button(
            "Download complete multi-task benchmark (CSV)", benchmark_csv_bytes(frame, dataset_key),
            file_name="MTL_benchmark.csv", mime="text/csv", key=f"{prefix}_download_full", use_container_width=True,
        )
    else:
        full_col, filtered_col = st.columns(2, gap="medium")
        with full_col:
            st.download_button(
                f"Download complete dataset · {len(frame):,} rows", benchmark_csv_bytes(frame, dataset_key),
                file_name=f"{dataset_key}_benchmark.csv", mime="text/csv", key=f"{prefix}_download_full", use_container_width=True,
            )
        with filtered_col:
            st.download_button(
                f"Download search results · {len(filtered):,} rows", benchmark_csv_bytes(filtered, dataset_key),
                file_name=f"{dataset_key}_filtered.csv", mime="text/csv", key=f"{prefix}_download_filtered", disabled=filtered.empty,
                use_container_width=True,
            )
    with st.expander("Dataset fields and preparation notes"):
        st.markdown(dataset_readme(frame, dataset_key))
        st.download_button(
            "Download dataset notes (Markdown)", dataset_readme(frame, dataset_key).encode("utf-8"),
            file_name=f"{dataset_key}_README.md", mime="text/markdown", key=f"{prefix}_download_notes",
        )


def render_data_resources_page(app_dir: Path) -> None:
    data_dir = Path(app_dir) / "data" / "benchmarks"
    st.markdown(CSS, unsafe_allow_html=True)
    _, center, _ = st.columns([1, 8, 1])
    with center, st.container(key="resources_page"):
        st.markdown(
            '<div class="resource-hero">'
            '<div class="resource-title">Data Resources</div>'
            '<div class="resource-intro">Explore the curated benchmarks supporting FXR and multi-target prediction. '
            'Review label distributions, browse molecular records and download the datasets for further research.</div>'
            '<span class="resource-tag">NURA &amp; ChEMBL</span>'
            '<span class="resource-tag">Searchable records</span>'
            '<span class="resource-tag">Source-free CSV downloads</span></div>',
            unsafe_allow_html=True,
        )
        descriptor_file = data_dir / "descriptor_cache.csv"
        cache = _cached_descriptors(str(data_dir), _signature(descriptor_file)) if descriptor_file.is_file() else None
        tabs = st.tabs(["FXR Agonist", "FXR Antagonist", "Multi-task"])
        for dataset_key, tab in zip(DATASETS, tabs):
            with tab:
                try:
                    frame = _cached_benchmark(str(data_dir), dataset_key, _signature(data_dir / f"{dataset_key}.csv"))
                except (OSError, pd.errors.ParserError, BenchmarkDataError) as error:
                    st.error(f"This benchmark could not be loaded: {error}")
                    continue
                filtered = _browse(frame, dataset_key)
                _overview(frame, dataset_key, cache)
                _downloads(frame, dataset_key, filtered)
