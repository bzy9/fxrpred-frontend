"""Integrated FXR resource page; all chemistry is precomputed offline."""
from html import escape
from pathlib import Path
import hashlib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data_resources import (
    RESOURCE_PROPERTIES, RESOURCE_LABELS, BenchmarkDataError,
    load_fxr_resources, filter_fxr_resources, resource_csv_bytes,
    scaffold_summary, load_benchmark_data, benchmark_csv_bytes,
    resource_preview, active_resource_groups,
)
PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
CSS = '\n<style>\n[data-testid="stAppViewContainer"]:has(.st-key-resources_page) { background: #f8fafc !important; }\n.st-key-resources_page { padding-bottom: 50px; }\n.resource-hero {\n    background: linear-gradient(135deg, #E0E7FF 0%, #F8FAFC 50%, #E0F2FE 100%);\n    border-radius: 16px; padding: 30px;\n    box-shadow: 0 20px 40px rgba(79,70,229,.05), 0 1px 3px rgba(0,0,0,.03);\n    margin-top: 20px; margin-bottom: 40px;\n    border: 1px solid rgba(79,70,229,.1);\n}\n.resource-title { font-size: 38px; font-weight: 900; color: #1e293b; margin-bottom: 15px; }\n.resource-intro { color: #475569; font-size: 16px; line-height: 1.8; }\n.resource-tag {\n    display: inline-block; background-color: white; padding: 6px 18px; border-radius: 20px;\n    font-size: 14px; font-weight: 700; color: #3b82f6;\n    box-shadow: 0 4px 6px rgba(0,0,0,.02); margin-right: 12px; margin-top: 20px;\n    border: 1px solid #bfdbfe;\n}\n.resource-metric {\n    background: #fff; border: 1px solid #e2e8f0; border-radius: 13px;\n    padding: 19px 17px; margin: 0 0 8px; min-height: 107px;\n    box-shadow: 0 4px 12px rgba(15,23,42,.025);\n}\n.resource-metric-value { font-size: 28px; font-weight: 850; color: #2563eb; line-height: 1.2; }\n.resource-metric-label { color: #64748b; font-size: 12px; margin-top: 10px; line-height: 1.5; }\n.resource-section {\n    font-size: 15px; font-weight: 700; display: block;\n    padding: 8px 16px; border-radius: 8px;\n    margin-top: 10px; margin-bottom: 15px;\n}\n.resource-section-query { background-color: #E0F2FE; color: #075985; border-left: 5px solid #0284C7; }\n.resource-section-overview { background-color: #DCFCE7; color: #166534; border-left: 5px solid #16A34A; }\n.resource-section-downloads { background-color: #E6FFFA; color: #0D9488; border-left: 5px solid #0D9488; }\n.resource-chart-title { font-size: 14px; font-weight: 750; color: #334155; margin: 12px 0 3px; }\n.resource-results { color: #475569; font-size: 13px; line-height: 1.7; margin: 17px 0 13px; }\n.st-key-resources_page [data-testid="stVerticalBlock"] { gap: .35rem !important; }\n.st-key-resources_page [data-baseweb="tab-list"] { gap: 12px; border-bottom: 1px solid #e2e8f0; }\n.st-key-resources_page [data-baseweb="tab"] { font-weight: 700; padding-left: 13px; padding-right: 13px; }\n.st-key-resources_page [data-testid="stDownloadButton"] button {\n    background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe;\n    border-radius: 9px; min-height: 44px; margin-top: 8px; font-weight: 650;\n}\n.st-key-resources_page [data-testid="stDownloadButton"] button:hover { background: #dbeafe; border-color: #93c5fd; }\n.st-key-resources_page [data-testid="stButton"] button { border-radius: 8px; }\n.st-key-resources_page [data-testid="stFormSubmitButton"] button {\n    background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%) !important;\n    color: white !important; border: none !important; border-radius: 8px !important;\n    box-shadow: 0 4px 10px rgba(79,70,229,.3) !important;\n    font-size: 16px !important; font-weight: bold !important; padding: 10px 0 !important;\n}\n.st-key-resources_page [data-testid="stFormSubmitButton"] button:hover {\n    transform: translateY(-2px); box-shadow: 0 6px 15px rgba(79,70,229,.4) !important;\n}\n@media (max-width: 768px) {\n    .resource-metric { padding: 15px 12px; }\n    .resource-metric-value { font-size: 24px; }\n}\n</style>\n'


def _signature(root):
    return tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(root.glob("*")) if p.is_file())


@st.cache_data(show_spinner=False)
def _resources(path, signature):
    return load_fxr_resources(Path(path))


@st.cache_data(show_spinner=False)
def _benchmark(path, name, signature):
    return load_benchmark_data(Path(path), name)


def _section(title, variant="overview"):
    st.markdown(f'<div class="resource-section resource-section-{variant}" role="heading" aria-level="2">{escape(title)}</div>', unsafe_allow_html=True)


def _metric(value, label):
    st.markdown('<div class="resource-metric">'+f'<div class="resource-metric-value">{escape(value)}</div>'+f'<div class="resource-metric-label">{escape(label)}</div></div>', unsafe_allow_html=True)


def _style(fig, height=280):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10,r=15,t=20,b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Arial, sans-serif",size=12,color="#475569"), showlegend=False)
    fig.update_xaxes(gridcolor="#edf2f7",zeroline=False)
    fig.update_yaxes(gridcolor="#edf2f7",zeroline=False)
    return fig


def _title(text):
    st.markdown(f'<div class="resource-chart-title">{escape(text)}</div>',unsafe_allow_html=True)


def _query(frame):
    _section("Data query", "query")
    with st.form("fxr_collection_search_v2", border=False):
        text_col, button_col = st.columns([6,1],gap="medium",vertical_alignment="bottom")
        with text_col:
            text = st.text_input("Search resources",placeholder="SMILES, Resource ID or Source ID",help="Enter a Resource ID number for an exact match. Literal SMILES text is case-sensitive; source IDs ignore case.",key="fxr_collection_query_text_v2")
        with button_col:
            submit = st.form_submit_button("Search",type="primary",use_container_width=True)
    if submit:
        st.session_state["fxr_collection_submitted_query_v2"] = text.strip()
    query = st.session_state.get("fxr_collection_submitted_query_v2", "")
    filtered = filter_fxr_resources(frame,query)
    cols = st.columns(3,gap="medium")
    active = int(filtered[list(RESOURCE_LABELS)].eq("1").any(axis=1).sum())
    metrics = [(f"{len(filtered):,}", "Matching molecules" if query else "Total molecules"),
               (f"{active:,}", "Active"),
               (f"{len(filtered)-active:,}", "Inactive")]
    for col,(value,label) in zip(cols,metrics):
        with col:_metric(value,label)
    view = resource_preview(filtered, query, limit=20)
    preview_text = f"showing the first {len(view):,} search matches" if query else f"previewing {len(view):,} fully annotated active molecules"
    st.markdown(f'<div class="resource-results"><b>{len(filtered):,}</b> matching molecules out of {len(frame):,} · {preview_text}</div>',unsafe_allow_html=True)
    if filtered.empty:
        st.info("No matching records. Clear the search text and click Search to restore the collection.")
        return filtered
    if view.empty:
        st.info("No active records with complete task annotations and source IDs are available for the default preview. Search to inspect individual records.")
        return filtered
    preview=view[["Resource ID","smiles","Activity annotations","molecular_weight","clogp","tpsa","source_ids"]].copy()
    preview["Resource ID"] = pd.to_numeric(preview["Resource ID"])
    event=st.dataframe(preview,hide_index=True,use_container_width=True,height=min(560,45+35*len(view)),on_select="rerun",selection_mode="single-row",key="fxr_collection_table_v3_"+hashlib.sha256(query.encode()).hexdigest()[:12],column_config={"Resource ID":st.column_config.NumberColumn("Resource ID",format="%d"),"smiles":st.column_config.TextColumn("SMILES",width="large"),"source_ids":st.column_config.TextColumn("Source IDs",width="large"),"molecular_weight":st.column_config.NumberColumn("MW",format="%.2f"),"clogp":st.column_config.NumberColumn("LogP",format="%.2f"),"tpsa":st.column_config.NumberColumn("TPSA",format="%.2f")})
    selected=event.get("selection",{}).get("rows",[]) if event else []
    if selected and selected[0]<len(view):
        r=view.iloc[selected[0]]
        with st.expander("Record details · "+r["Resource ID"],expanded=True):
            st.code(r["smiles"],language=None)
            task_rows=[]
            for column,positive,negative in [(RESOURCE_LABELS[0],"Agonist (1)","Non-agonist (0)"),(RESOURCE_LABELS[1],"Antagonist (1)","Non-antagonist (0)")]:
                task_rows.append({"Task":"FXR agonist" if column==RESOURCE_LABELS[0] else "FXR antagonist","Annotation":{"1":positive,"0":negative,"":"Not annotated"}[r[column]]})
            st.dataframe(pd.DataFrame(task_rows),hide_index=True,use_container_width=True)
            st.dataframe(pd.DataFrame([{"Property":title,"Value":round(float(r[key]),3)} for key,(title,_) in RESOURCE_PROPERTIES.items()]),hide_index=True,use_container_width=True)
            st.text("Source IDs: "+(r["source_ids"] or "Not available"))
    return filtered


def _profiles(frame, key_prefix="fxr", color="#60a5fa"):
    _section("Molecular Property Profiles")
    if frame.empty:return
    for start in (0,3):
        cols=st.columns(3,gap="medium")
        for col,(key,(title,axis)) in zip(cols,list(RESOURCE_PROPERTIES.items())[start:start+3]):
            with col:
                _title(title)
                params={"xbins":dict(start=-.5,end=float(frame[key].max())+.5,size=1)} if key in ("hba","hbd") else {"nbinsx":35}
                fig=go.Figure(go.Histogram(x=frame[key],histnorm="percent",marker_color=color,hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",**params))
                fig.update_xaxes(title_text=axis);fig.update_yaxes(title_text="Molecules (%)")
                st.plotly_chart(_style(fig,245),use_container_width=True,config=PLOT_CONFIG,key=key_prefix+"_property_"+key)


def _structure_profiles(frame, key_prefix="fxr", color="#14b8a6"):
    _section("Physicochemical Space & Scaffolds")
    if frame.empty:return
    left,right=st.columns([1.15,1],gap="large")
    with left:
        _title("Molecular weight and lipophilicity")
        fig=go.Figure(go.Scattergl(x=frame["molecular_weight"],y=frame["clogp"],mode="markers",customdata=frame[["Resource ID","tpsa"]].to_numpy(),marker=dict(size=6,opacity=.65,color=frame["tpsa"],colorscale="Viridis",showscale=True,colorbar=dict(title="TPSA",thickness=12)),hovertemplate="%{customdata[0]}<br>MW: %{x:.2f}<br>LogP: %{y:.2f}<br>TPSA: %{customdata[1]:.2f}<extra></extra>"))
        fig.update_xaxes(title_text="Molecular weight (g/mol)");fig.update_yaxes(title_text="Calculated LogP")
        st.plotly_chart(_style(fig,330),use_container_width=True,config=PLOT_CONFIG,key=key_prefix+"_property_space")
    with right:
        _title("Most frequent ring/linker scaffolds")
        summary=scaffold_summary(frame)
        if summary.empty:st.info("No ring/linker scaffold is present in the current selection.")
        else:
            fig=go.Figure(go.Bar(x=summary["Molecules"],y=summary["Scaffold"],orientation="h",marker_color=color,customdata=summary[["Scaffold SMILES","Fraction (%)"]].to_numpy(),text=summary["Molecules"],textposition="auto",hovertemplate="%{y}<br>%{customdata[0]}<br>%{x} molecules (%{customdata[1]:.1f}%)<extra></extra>"))
            fig.update_yaxes(autorange="reversed");fig.update_xaxes(title_text="Molecules")
            st.plotly_chart(_style(fig,330),use_container_width=True,config=PLOT_CONFIG,key=key_prefix+"_scaffold_frequency")
    with st.expander("Scaffold key and calculation notes"):
        st.dataframe(scaffold_summary(frame),hide_index=True,use_container_width=True)
        st.write(f"Acyclic molecules: {int(frame['scaffold_smiles'].eq('').sum()):,}. Scaffold percentages use all matching molecules as the denominator.")
        st.write("Bemis–Murcko scaffolds summarize ring and linker frameworks; they do not establish activity-determining fragments. Molecular properties use the largest organic component; the original full SMILES is retained.")



def _active_visualizations(frame):
    groups = active_resource_groups(frame)
    tabs = st.tabs([
        f"FXR agonists · {len(groups['AGO']):,}",
        f"FXR antagonists · {len(groups['ANT']):,}",
    ])
    for tab, task, color in zip(tabs, ("AGO", "ANT"), ("#3b82f6", "#14b8a6")):
        with tab:
            active_frame = groups[task]
            if active_frame.empty:
                st.info("No positive annotations for this task in the current search results.")
                continue
            prefix = "fxr_active_" + task.lower()
            _profiles(active_frame, key_prefix=prefix, color=color)
            _structure_profiles(active_frame, key_prefix=prefix, color=color)


def _downloads(frame,filtered,app_dir):
    _section("Downloads","downloads")
    full,results=st.columns(2,gap="medium")
    with full:st.download_button(f"Complete FXR collection · {len(frame):,} molecules",resource_csv_bytes(frame),file_name="FXR_activity_resources.csv",mime="text/csv",key="fxr_resources_download_all",use_container_width=True)
    with results:st.download_button(f"Search results · {len(filtered):,} molecules",resource_csv_bytes(filtered),file_name="FXR_activity_search_results.csv",mime="text/csv",key="fxr_resources_download_search",disabled=filtered.empty,use_container_width=True)
    _title("Model benchmark datasets")
    cols=st.columns(3,gap="medium")
    root=Path(app_dir)/"data/benchmarks"
    for col,key,title in zip(cols,("FXR_AGO","FXR_ANT","MTL"),("FXR agonist benchmark","FXR antagonist benchmark","Complete multi-task benchmark")):
        with col:
            try:
                data=_benchmark(str(root),key,_signature(root))
                st.download_button(f"{title} · {len(data):,} rows",benchmark_csv_bytes(data,key),file_name=key+"_benchmark.csv",mime="text/csv",key="fxr_benchmark_download_"+key,use_container_width=True)
            except (OSError,ValueError,KeyError) as exc:st.error(f"{title} could not be loaded: {exc}")
    with st.expander("Dataset fields and preparation notes"):
        st.markdown("The FXR collection combines positive and negative annotations from the curated FXR benchmarks and additional external records. Records are merged by canonical isomeric SMILES. Agonist and antagonist labels are retained separately; an unannotated task is not a negative observation. Inactive denotes no positive annotation among the available FXR tasks; it does not imply that both tasks were tested.")
        st.markdown("The default preview prioritizes active molecules with complete task annotations, source IDs and molecular properties. Search and downloads cover the full collection. Charts use the displayed positive labels for each task. For the reviewed partial-agonist entry, the display follows the existing FXR agonist benchmark; the original external annotation is retained in the source records.")
        st.markdown("Collection CSVs contain Resource ID, SMILES, activity annotations, two FXR task labels and six molecular properties. Binary benchmark downloads retain smiles and label. The complete multi-task benchmark retains smiles, group and all four task labels, including missing values and original dataset assignments.")


def render_data_resources_page(app_dir):
    st.markdown(CSS,unsafe_allow_html=True)
    _,center,_=st.columns([1,8,1])
    with center,st.container(key="resources_page"):
        st.markdown('<div class="resource-hero"><div class="resource-title">Data Resources</div><div class="resource-intro">Explore the curated FXR activity dataset. Search molecular records, inspect structural and physicochemical profiles, and download resources for further research.</div><span class="resource-tag">FXR activity data</span><span class="resource-tag">Searchable records</span><span class="resource-tag">Molecular profiles</span></div>',unsafe_allow_html=True)
        root=Path(app_dir)/"data/fxr_resources"
        try:frame=_resources(str(root),_signature(root))
        except (OSError,ValueError,KeyError) as exc:
            st.error(f"The FXR collection could not be loaded: {exc}")
            return
        filtered=_query(frame)
        _active_visualizations(filtered)
        _downloads(frame,filtered,app_dir)
