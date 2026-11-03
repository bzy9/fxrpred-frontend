
import streamlit as st
import pandas as pd
import os
import base64
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

import api_client
from data_resources_page import render_data_resources_page



def prepare_ortho_csv_input(df):
    """用同一行筛选 SMILES 和 ID；CSV 应以字符串读取，保留 ID 前导零。"""
    smiles_cols = [col for col in df.columns if str(col).strip().lower() == "smiles"]
    id_cols = [col for col in df.columns if str(col).strip().lower() == "np_id"]
    if len(smiles_cols) != 1:
        raise ValueError("CSV must contain exactly one SMILES column (case-insensitive).")
    if len(id_cols) > 1:
        raise ValueError("CSV contains multiple np_id columns.")
    smiles = df[smiles_cols[0]].fillna("").astype(str).str.strip()
    keep = smiles.ne("")
    ids = (df[id_cols[0]].fillna("").astype(str)
           if id_cols else pd.Series("", index=df.index))
    return smiles[keep].tolist(), ids[keep].tolist(), int((~keep).sum())


def validate_ortho_results(api_response, smiles_list, np_ids):
    """只有完整、顺序对应且概率有效的结果才允许展示和下载。"""
    if not isinstance(api_response, dict):
        raise ValueError("Backend returned an invalid response format.")
    if api_response.get("error"):
        raise RuntimeError(str(api_response["error"]))
    results = api_response.get("results_list")
    if not isinstance(results, list) or not results:
        raise ValueError("Backend returned no prediction results.")
    if len(results) != len(smiles_list) or len(np_ids) != len(smiles_list):
        raise ValueError("Result count does not match input count; download was blocked to prevent ID mismatch.")
    tasks = ["FXR_ago", "PXR_ago", "LXR_ago", "PPAR_ago"]
    for index, (result, smiles) in enumerate(zip(results, smiles_list), start=1):
        if not isinstance(result, dict) or result.get("SMILES") != smiles:
            raise ValueError(f"SMILES order mismatch at result {index}; download was blocked.")
        for task in tasks:
            if result.get(task) not in ("Active", "Inactive"):
                raise ValueError(f"Missing or invalid {task} label at result {index}.")
            prob = result.get(f"{task}_prob")
            if isinstance(prob, bool) or not isinstance(prob, (int, float)) or not 0.0 <= prob <= 1.0:
                raise ValueError(f"Missing or invalid {task}_prob at result {index}; check the updated backend.")
    frame = pd.DataFrame(results)
    frame.insert(0, "np_id", np_ids)
    columns = ["np_id", "SMILES"]
    for task in tasks:
        columns.extend([task, f"{task}_prob"])
    return frame[columns]


APP_DIR = Path(__file__).resolve().parent


# ==========================================
# 🖼️ 1. 安全读取本地背景图片逻辑
# ==========================================
def get_base64_of_bin_file(bin_file):
    """读取本地文件并转换为 Base64 编码，若找不到文件则返回空"""
    bin_file = APP_DIR / bin_file
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

# 读取背景图 (若没找到则为空字符串)
hero_img_base64 = get_base64_of_bin_file("assets/hero_bg.png")

# 构建安全的 CSS 背景属性
if hero_img_base64:
    hero_bg_css = f'url("data:image/png;base64,{hero_img_base64}")'
else:
    hero_bg_css = 'none'

# ==========================================
# 🎨 2. FXRPred 全局 CSS
# ==========================================
css_code = """
<style>
/* 🚀 引入现代几何感字体 Poppins */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;800;900&display=swap');

/* =====================================================
   1. 全局字体与重置
   ===================================================== */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif !important;
}
header[data-testid="stHeader"], [data-testid="collapsedControl"], #MainMenu, footer {
    display: none !important;
}
[data-testid="stAppViewContainer"] { background: #ffffff !important; }
[data-testid="stAppViewContainer"] > section.main { padding-top: 0 !important; }
.block-container { max-width: 100% !important; padding: 0 !important; margin: 0 !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }

/* =====================================================
   2. 顶部导航栏 & Logo (🌟 导航栏字母渐变与对齐)
   ===================================================== */
.fxr-navbar {
    width: 100%; height: 78px; background: rgba(255, 255, 255, 0.98);
    border-bottom: 1px solid #e5e7eb; display: flex; align-items: center;
    padding: 0 48px; box-sizing: border-box; position: relative; z-index: 999;
}
.fxr-logo-wrapper { display: flex; align-items: center; min-width: 250px; height: 78px; margin-left: 10%; }
.fxr-logo-icon { width: 42px; height: 42px; margin-right: 12px; display: flex; align-items: center; justify-content: center; }
.fxr-logo-icon svg { width: 42px; height: 42px; }

/* 🚀 导航栏文字：统一使用 Poppins 字体，与主页大标题同款渐变色 */
.fxr-logo-text { 
    font-size: 29px; 
    font-weight: 800; 
    letter-spacing: -0.5px; 
    white-space: nowrap; 
    background: linear-gradient(90deg, #1e3a8a 0%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* =====================================================
   3. 导航按钮 
   ===================================================== */
div[data-testid="stHorizontalBlock"]:has(.fxr-logo-wrapper) div.stButton { display: flex; align-items: center; justify-content: center; }
div[data-testid="stHorizontalBlock"]:has(.fxr-logo-wrapper) div.stButton > button {
    border: none !important; background: transparent !important; color: #64748b !important; /* 默认浅灰色 */
    font-size: 16px !important; font-weight: 500 !important; height: 48px !important;
    padding: 0 16px !important; border-radius: 0 !important; box-shadow: none !important;
    transition: all 0.2s ease !important; position: relative; white-space: nowrap;
    margin-top: 18px !important; 
}
div[data-testid="stHorizontalBlock"]:has(.fxr-logo-wrapper) div.stButton > button:hover { color: #2563eb !important; }

/* =====================================================
   4. Hero 区域 (🌟 修复：标题变大，间距优化)
   ===================================================== */
.fxr-hero {
    width: 100%; height: 380px; position: relative; overflow: hidden;
    background-color: #f7fbff; background-image: HERO_BG_PLACEHOLDER;
    background-size: cover; background-position: center; background-repeat: no-repeat;
}

.fxr-hero-content {
    max-width: 1360px; height: 100%; margin: 0 auto; 
    padding: 35px 42px 0 12%; 
    box-sizing: border-box; display: flex; flex-direction: column; justify-content: flex-start;
}
.fxr-hero-left { width: 65%; position: relative; z-index: 5; }

/* 字体放大到 100px，letter-spacing(字间距) 设置为 8px，你可以根据喜好微调 */
.fxr-hero-title { font-size: 80px !important; line-height: 1.05; font-weight: 900; letter-spacing: 0.5px; margin: 0 0 -10px 0 !important; color: #0f172a; }
.fxr-hero-title span { 
    background: linear-gradient(90deg, #1e3a8a 0%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.fxr-hero-subtitle { font-size: 24px; line-height: 1.3; font-weight: 600; color: #1e293b; margin-bottom: 20px; }
.fxr-hero-line { width: 58px; height: 4px; background: linear-gradient(90deg, #2563eb, #38bdf8); border-radius: 5px; margin-bottom: 20px; }
.fxr-hero-description { max-width: 650px; font-size: 16px; line-height: 1.7; color: #475569; margin-bottom: 0px; }

/* =====================================================
   5. Hero CTA 按钮 
   ===================================================== */
div[data-testid="stHorizontalBlock"]:has(#hero-btn-marker) { margin-top: -80px !important; position: relative; z-index: 20; }

div[data-testid="stHorizontalBlock"]:has(#hero-btn-marker) > div:nth-child(2) button {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important; color: white !important; border: none !important; 
    font-size: 16px !important; font-weight: 600 !important; padding: 8px 15px !important; border-radius: 8px !important; 
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
}
div[data-testid="stHorizontalBlock"]:has(#hero-btn-marker) > div:nth-child(2) button:hover { transform: translateY(-2px) !important; }

div[data-testid="stHorizontalBlock"]:has(#hero-btn-marker) > div:nth-child(3) button {
    background: white !important; color: #2563eb !important; border: 2px solid #93c5fd !important; 
    font-size: 16px !important; font-weight: 600 !important; padding: 8px 15px !important; border-radius: 8px !important;
}
div[data-testid="stHorizontalBlock"]:has(#hero-btn-marker) > div:nth-child(3) button:hover { background: #f0f7ff !important; border-color: #3b82f6 !important; }

/* =====================================================
   6. 功能卡片区域 (🌟 恢复列表排版 + 3D 渐变渲染)
   ===================================================== */
.fxr-features-section { padding: 0 0 30px 0; }
.fxr-feature-card {
    height: 300px; /* 保持等高 */
    border-radius: 16px; 
    padding: 30px; box-sizing: border-box; 
    /* 🚀 核心大招：外阴影增加立体悬浮感，内阴影 (inset) 增加顶部边缘高光反光，拉满 3D 质感 */
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.04), inset 0 2px 4px rgba(255, 255, 255, 0.9);
    transition: transform 0.3s ease, box-shadow 0.3s ease; 
    display: flex; flex-direction: column;
    border: 1px solid rgba(255, 255, 255, 0.6) !important; 
}
.fxr-feature-card:hover { 
    transform: translateY(-5px); 
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.08), inset 0 2px 4px rgba(255, 255, 255, 0.9); 
}

/* 🚀 专属 3D 对角线平滑渐变背景 */
.card-blue { background: linear-gradient(135deg, #ffffff 0%, #e0f2fe 100%); }
.card-green { background: linear-gradient(135deg, #ffffff 0%, #dcfce7 100%); }
.card-purple { background: linear-gradient(135deg, #ffffff 0%, #f3e8ff 100%); }

/* 恢复图标并增加微投影 */
.feature-icon { width: 48px; height: 48px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 22px; margin-bottom: 16px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
.feature-icon-blue { background: #eff6ff; color: #3b82f6; border: 1px solid #bfdbfe; }
.feature-icon-green { background: #f0fdf4; color: #22c55e; border: 1px solid #bbf7d0; }
.feature-icon-purple { background: #f5f3ff; color: #8b5cf6; border: 1px solid #e9d5ff; }

/* 恢复文本层级与样式 */
.feature-title { font-size: 21px; font-weight: 800; margin-bottom: 6px; }
.feature-title-blue { color: #1e3a8a; }
.feature-title-green { color: #14532d; }
.feature-title-purple { color: #4c1d95; }

.feature-subtitle { font-size: 14px; color: #64748b; margin-bottom: 15px; }

/* 恢复列表形式 */
.feature-list { padding: 0; margin: 0; list-style: none; font-size: 14px; line-height: 2.1; color: #475569; }
.feature-list li::before { content: "✓"; margin-right: 8px; font-weight: bold; font-size: 14px; }
.feature-list-blue li::before { color: #3b82f6; }
.feature-list-green li::before { color: #22c55e; }
.feature-list-purple li::before { color: #8b5cf6; }

/* =====================================================
   7. 底部统计栏
   ===================================================== */
.fxr-stats { max-width: 1360px; margin: 0 auto; padding: 20px 42px 50px 42px; }
.stats-container {
    border: 1px solid #e2e8f0; border-radius: 16px; background: #ffffff;
    min-height: 120px; display: flex; align-items: center; justify-content: space-around;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02); padding: 20px 0;
}
.stat-item { flex: 1; text-align: center; position: relative; }
.stat-item:not(:last-child)::after { content: ""; position: absolute; right: 0; top: 15px; height: 60px; width: 1px; background: #e2e8f0; }
.stat-icon { font-size: 28px; color: #3b82f6; margin-bottom: 8px; }
.stat-number { font-size: 26px; font-weight: 800; color: #0f172a; line-height: 1.2; }
.stat-label { font-size: 13px; color: #64748b; margin-top: 4px; font-weight: 500; }
</style>
"""

css_code = css_code.replace("HERO_BG_PLACEHOLDER", hero_bg_css)
st.markdown(css_code, unsafe_allow_html=True)

# ==========================================
# 🚀 3. 顶部导航栏渲染
# ==========================================
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# Migrate existing browser sessions to the renamed resource page.
if st.session_state.current_page == "Substructure Analysis":
    st.session_state.current_page = "Data Resources"

nav_pages = ["Home", "FXR Prediction", "Multi-target Prediction", "Data Resources", "Help"]
nav_cols = st.columns([2.7, 1.05, 1.45, 1.75, 1.65, 0.75], gap="small")

# 1. 在这里读取你的 Logo 图片
logo_base64 = get_base64_of_bin_file("assets/logo.png")
logo_html = f'<img src="data:image/png;base64,{logo_base64}" style="width: 42px; height: 42px; object-fit: contain;">' if logo_base64 else ''

with nav_cols[0]:
    st.markdown(
        f"""
        <div class="fxr-logo-wrapper">
            <div class="fxr-logo-icon">
                {logo_html}
            </div>
            <div class="fxr-logo-text">FXR<span>Pred</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )


page = st.session_state.current_page

# 🌟 动态注入 CSS，高亮当前所在的页面按钮
# 因为 logo 占了第 1 列 (索引为1)，所以右边的按钮从第 2 列开始算。
if page in nav_pages:
    active_index = nav_pages.index(page) + 2
    st.markdown(f"""
        <style>
        div[data-testid="stHorizontalBlock"]:has(.fxr-logo-wrapper) > div:nth-child({active_index}) button {{
            color: #1e3a8a !important;
            font-weight: 800 !important;
            border-bottom: 3px solid #3b82f6 !important;
        }}
        </style>
    """, unsafe_allow_html=True)

# 渲染按钮
for i, label in enumerate(nav_pages, start=1):
    with nav_cols[i]:
        if st.button(label, key=f"nav_{label}", use_container_width=True):
            st.session_state.current_page = label
            st.rerun()

page = st.session_state.current_page

# ============================================================
# 🏠 4. HOME PAGE 主体内容
# ============================================================
if page == "Home":

    # --- 渲染 Hero 横幅 ---
    st.markdown(
        """
        <div class="fxr-hero">
            <div class="fxr-hero-content">
                <div class="fxr-hero-left">
                    <h1 class="fxr-hero-title">FXR<span>Pred</span></h1>
                    <div class="fxr-hero-subtitle">A Deep Learning Platform for FXR Modulators</div>
                    <div class="fxr-hero-line"></div>
                    <div class="fxr-hero-description">
                        FXRPred integrates deep learning models to predict FXR modulators and multi-target
                        activities, with searchable benchmark datasets supporting further research.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- 悬浮交互按钮 ---
    col_spacer, col_btn1, col_btn2, col_rest = st.columns([1.2, 1.8, 1.8, 5.2])

    with col_spacer:
        st.markdown('<span id="hero-btn-marker"></span>', unsafe_allow_html=True)

    with col_btn1:
        start_prediction = st.button("🔎 Start Prediction", use_container_width=True)

    with col_btn2:
        learn_more = st.button("📖 Learn More", use_container_width=True)

    st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)

    if start_prediction:
        st.session_state.current_page = "FXR Prediction"
        st.rerun()
    if learn_more:
        st.session_state.current_page = "Help"
        st.rerun()

    # --- 三大核心功能卡片 (优化版排版) ---
    st.markdown('<div class="fxr-features-section"><div class="fxr-feature-container">', unsafe_allow_html=True)
    card1, card2, card3 = st.columns([1, 1, 1], gap="large")

    icon1 = get_base64_of_bin_file("assets/icon_fxr.png")
    icon2 = get_base64_of_bin_file("assets/icon_moe.png")

    # 🌟 将图片尺寸稍微放大到 48px，以匹配右侧两行文字的高度
    img_tag1 = f'<img src="data:image/png;base64,{icon1}" style="width: 48px; height: 48px; object-fit: contain;">' if icon1 else ''
    img_tag2 = f'<img src="data:image/png;base64,{icon2}" style="width: 48px; height: 48px; object-fit: contain;">' if icon2 else ''
    img_tag3 = '''<svg width="48" height="48" viewBox="0 0 48 48" fill="none" aria-hidden="true">
        <rect x="1" y="1" width="46" height="46" rx="13" fill="#F5F3FF" stroke="#DDD6FE"/>
        <ellipse cx="24" cy="15" rx="11" ry="5" stroke="#8B5CF6" stroke-width="2"/>
        <path d="M13 15v18c0 2.8 4.9 5 11 5s11-2.2 11-5V15M13 24c0 2.8 4.9 5 11 5s11-2.2 11-5" stroke="#8B5CF6" stroke-width="2"/>
    </svg>'''

    with card1:
        st.markdown(
            f"""
                        <div class="fxr-feature-card card-blue">
                            <!-- 🌟 外层 Flex 横向排布 Logo 和 右侧文字区 -->
                            <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 15px;">
                                {img_tag1}
                                <!-- 🌟 内层 Flex 纵向排布标题和副标题 -->
                                <div style="display: flex; flex-direction: column; justify-content: center;">
                                    <div class="feature-title feature-title-blue" style="margin-bottom: 4px;">FXR Prediction</div>
                                    <div class="feature-subtitle" style="margin-bottom: 0;">High-performance binary classification</div>
                                </div>
                            </div>
                            <ul class="feature-list feature-list-blue">
                                <li>FXR Agonist vs. Non-agonist</li>
                                <li>FXR Antagonist vs. Non-antagonist</li>
                                <li>Attention-based interpretation</li>
                                <li>Key pharmacophore identification</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True
        )

    with card2:
        st.markdown(
            f"""
                        <div class="fxr-feature-card card-green">
                            <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 15px;">
                                {img_tag2}
                                <div style="display: flex; flex-direction: column; justify-content: center;">
                                    <div class="feature-title feature-title-green" style="margin-bottom: 4px;">Multi-target Prediction</div>
                                    <div class="feature-subtitle" style="margin-bottom: 0;">Joint prediction for multiple receptors</div>
                                </div>
                            </div>
                            <ul class="feature-list feature-list-green">
                                <li>FXR, PXR, LXR, PPARG agonists</li>
                                <li>Multi-task learning </li>
                                <li>Shared & target-specific knowledge</li>
                                <li>Unimol 3D structure</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True
        )

    with card3:
        st.markdown(
            f"""
                        <div class="fxr-feature-card card-purple">
                            <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 15px;">
                                {img_tag3}
                                <div style="display: flex; flex-direction: column; justify-content: center;">
                                    <div class="feature-title feature-title-purple" style="margin-bottom: 4px;">Data Resources</div>
                                    <div class="feature-subtitle" style="margin-bottom: 0;">Curated molecular benchmarks</div>
                                </div>
                            </div>
                            <ul class="feature-list feature-list-purple">
                                <li>FXR & multi-task benchmark datasets</li>
                                <li>Searchable SMILES and source IDs</li>
                                <li>Label & molecular property profiles</li>
                                <li>Source-free CSV downloads</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True
        )
    st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)
    # --- 统计数据区域 (专业 SVG 图标版) ---
    st.markdown(
        """
        <div class="fxr-stats">
            <div class="stats-container">
                <div class="stat-item">
                    <div class="stat-icon">
                        <!-- 化合物库 / 数据库 图标 -->
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>
                            <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path>
                            <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path>
                        </svg>
                    </div>
                    <div class="stat-number">10,000+</div>
                    <div class="stat-label">Compounds</div>
                </div>
                <div class="stat-item">
                    <div class="stat-icon">
                        <!-- 受体靶标 / 精准打击 图标 -->
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="10"></circle>
                            <circle cx="12" cy="12" r="6"></circle>
                            <circle cx="12" cy="12" r="2"></circle>
                        </svg>
                    </div>
                    <div class="stat-number">4</div>
                    <div class="stat-label">Targets</div>
                </div>
                <div class="stat-item">
                    <div class="stat-icon">
                        <!-- 深度学习 / 神经网络 图标 -->
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="18" cy="5" r="3"></circle>
                            <circle cx="6" cy="12" r="3"></circle>
                            <circle cx="18" cy="19" r="3"></circle>
                            <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
                            <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
                        </svg>
                    </div>
                    <div class="stat-number">2</div>
                    <div class="stat-label">Deep Learning Models</div>
                </div>
                <div class="stat-item">
                    <div class="stat-icon">
                        <!-- 3D 特征提取 / 空间几何 图标 -->
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                            <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                            <line x1="12" y1="22.08" x2="12" y2="12"></line>
                        </svg>
                    </div>
                    <div class="stat-number">3D</div>
                    <div class="stat-label">Feature extraction</div>
                </div>
                <div class="stat-item">
                    <div class="stat-icon">
                        <!-- 并发 / 闪电 图标 -->
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
                        </svg>
                    </div>
                    <div class="stat-number">High</div>
                    <div class="stat-label">Concurrency</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


elif page == "FXR Prediction":
    # ==========================================
    # 🎨 局部 CSS 注入：FACA-GIN 专属 3D 高级样式 (保持不变)
    # ==========================================
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #F8FAFC; 
    }

    .faca-hero-card {
        background: linear-gradient(135deg, #E0E7FF 0%, #F8FAFC 50%, #E0F2FE 100%);
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 20px 40px rgba(79, 70, 229, 0.05), 0 1px 3px rgba(0,0,0,0.03);
        margin-top: 20px;
        margin-bottom: 40px;
        border: 1px solid rgba(79, 70, 229, 0.1);
    }

    .faca-hero-title { font-size: 38px; font-weight: 900; color: #1e293b; margin-bottom: 15px; }
    .faca-hero-subtitle { color: #475569; font-size: 16px; line-height: 1.8; }

    .hero-pill {
        display: inline-block; background-color: white; padding: 6px 18px; border-radius: 20px;
        font-size: 14px; font-weight: 700; color: #3b82f6; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.02); margin-right: 12px; margin-top: 20px;
        border: 1px solid #bfdbfe;
    }

    .faca-prompt-model { font-size: 15px; font-weight: 700; display: block; padding: 8px 16px; border-radius: 8px; margin-top: 10px; margin-bottom: 15px; background-color: #E6FFFA; color: #0D9488; border-left: 5px solid #0D9488; }
    .faca-prompt-threshold { font-size: 15px; font-weight: 700; display: block; padding: 8px 16px; border-radius: 8px; margin-top: 10px; margin-bottom: 15px; background: linear-gradient(90deg, #EEF2FF 0%, #EFF6FF 100%); color: #3730A3; border-left: 5px solid #4F46E5; }
    .faca-prompt-data { font-size: 15px; font-weight: 700; display: block; padding: 8px 16px; border-radius: 8px; margin-top: 10px; margin-bottom: 15px; background-color: #E0F2FE; color: #075985; border-left: 5px solid #0284C7; }
    .faca-prompt-result { font-size: 15px; font-weight: 700; display: block; padding: 8px 16px; border-radius: 8px; margin-top: 10px; margin-bottom: 15px; background-color: #DCFCE7; color: #166534; border-left: 5px solid #16A34A; }

    .faca-threshold-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin: 4px 0 8px 0; }
    .faca-threshold-chip { display: inline-block; padding: 6px 12px; border-radius: 999px; background-color: #EEF2FF; border: 1px solid #C7D2FE; color: #4338CA; font-size: 13px; font-weight: 700; }
    .faca-threshold-chip-current { background-color: #EFF6FF; border-color: #BFDBFE; color: #1D4ED8; }
    .faca-threshold-note { color: #64748B; font-size: 13px; line-height: 1.6; margin-bottom: 12px; }

    [data-testid="stSlider"] { padding: 6px 4px 2px 4px; }
    [data-testid="stSlider"] [role="slider"] {
        background-color: #4F46E5 !important;
        border-color: white !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.18) !important;
    }

    .faca-status-info { font-size: 15px; color: #0369A1; background-color: #F0F9FF; padding: 12px 16px; border-radius: 8px; border: 1px solid #BAE6FD; margin-bottom: 15px; display: block; }
    .faca-status-success { font-size: 15px; color: #15803D; background-color: #F0FDF4; padding: 12px 16px; border-radius: 8px; border: 1px solid #BBF7D0; margin-bottom: 15px; display: block; }

    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%) !important; 
        color: white !important; border: none !important; border-radius: 8px !important;
        box-shadow: 0 4px 10px rgba(79, 70, 229, 0.3) !important; transition: all 0.3s ease !important;
        font-size: 16px !important; font-weight: bold !important; padding: 10px 0px !important;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="baseButton-primary"]:hover {
        transform: translateY(-2px) !important; box-shadow: 0 6px 15px rgba(79, 70, 229, 0.4) !important; color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    spacer_left, center_col, spacer_right = st.columns([1, 8, 1])

    with center_col:
        st.markdown("""
        <div class="faca-hero-card">
            <div class="faca-hero-title">FXR Agonist & Antagonist Prediction</div>
            <div class="faca-hero-subtitle">
                Input molecular data for automated activity prediction driven by fragment-based graph attention networks and ensemble inference.
            </div>
            <span class="hero-pill"> FACA-GIN</span>
            <span class="hero-pill"> BRICS Fragments</span>
            <span class="hero-pill"> Bidirectional Cross-Attention</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="faca-prompt-model">💡 Select Prediction Direction</div>', unsafe_allow_html=True)
        model_choice = st.radio(
            label="model_select_hidden",
            options=["🔴 FXR Agonist", "🔵 FXR Antagonist"],
            horizontal=True,
            label_visibility="collapsed"
        )

        # 前端只传模型标识，权重位置由后端管理。
        if "Agonist" in model_choice:
            model_key = "AGO"
        else:
            model_key = "ANT"

        threshold_config_key = f"faca_{model_key}_threshold_config"
        threshold_widget_key = f"faca_{model_key}_activity_threshold"

        # 同一会话中每个模型只查询一次服务器配置，避免 Streamlit 重运行时重复请求。
        if threshold_config_key not in st.session_state:
            try:
                threshold_config = api_client.get_faca_recommended_threshold(model_key)
                recommended_threshold = float(threshold_config["recommended_threshold"])
                if not 0.0 <= recommended_threshold <= 1.0:
                    raise ValueError("Recommended threshold is outside the valid range.")

                st.session_state[threshold_config_key] = {
                    "recommended_threshold": recommended_threshold,
                    "source": threshold_config.get("source", "cross_validation_summary.json"),
                    "used_fallback": threshold_config.get("source") == "fallback_0.5"
                }
            except Exception:
                st.session_state[threshold_config_key] = {
                    "recommended_threshold": 0.5,
                    "source": "fallback_0.5",
                    "used_fallback": True
                }

        threshold_config = st.session_state[threshold_config_key]
        recommended_threshold = float(threshold_config["recommended_threshold"])

        if threshold_widget_key not in st.session_state:
            st.session_state[threshold_widget_key] = recommended_threshold

        st.markdown(
            '<div class="faca-prompt-threshold">⚙️ Activity Decision Threshold</div>',
            unsafe_allow_html=True
        )

        reset_threshold = st.button(
            "↺ Reset to Model-Recommended Threshold",
            key=f"reset_faca_{model_key}_threshold"
        )
        if reset_threshold:
            st.session_state[threshold_widget_key] = recommended_threshold

        activity_threshold = st.slider(
            label="Activity decision threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format="%.2f",
            key=threshold_widget_key,
            label_visibility="collapsed",
            help="Molecules with a predicted probability greater than or equal to this value are classified as active."
        )

        st.markdown(
            f"""
            <div class="faca-threshold-meta">
                <span class="faca-threshold-chip">Model-recommended: {recommended_threshold:.2f}</span>
                <span class="faca-threshold-chip faca-threshold-chip-current">Current: {activity_threshold:.2f}</span>
            </div>
            <div class="faca-threshold-note">
                Molecules with predicted probability greater than or equal to the current threshold are classified as active.
                A higher threshold applies a stricter activity criterion.
            </div>
            """,
            unsafe_allow_html=True
        )

        if threshold_config["used_fallback"]:
            st.caption(
                "⚠️ The model-recommended threshold could not be read from the server, so 0.50 is being used as the fallback default."
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="faca-prompt-data">🧬 Input Molecular Data</div>', unsafe_allow_html=True)
        input_method = st.radio(
            label="input_method_hidden",
            options=["✍️ Input SMILES", "📁 Upload CSV File"],
            horizontal=True,
            label_visibility="collapsed"
        )

        smiles_list = []
        if input_method == "✍️ Input SMILES":
            raw_smiles = st.text_area(
                label="smiles_input_hidden",
                value="",
                height=150,
                placeholder="Paste SMILES strings here, use Enter for multiple lines...\nExample: CN(C)C(=O)c1cccc(-c2ccc(OCc3c(-c4c(Cl)cccc4Cl)noc3C3CC3)cc2)c1",
                label_visibility="collapsed"
            )
            if raw_smiles:
                smiles_list = [s.strip() for s in raw_smiles.split('\n') if s.strip()]
        else:
            uploaded_file = st.file_uploader(label="csv_upload_hidden", type=["csv"], label_visibility="collapsed")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
                    st.markdown('<div class="faca-status-success">✅ File uploaded successfully!</div>',
                                unsafe_allow_html=True)
                    with st.expander("Click to view data preview"):
                        st.dataframe(df.head(), use_container_width=True)

                    target_col = next((col for col in df.columns if str(col).lower() == 'smiles'), None)
                    if target_col is not None:
                        smiles_list = [s.strip() for s in df[target_col].astype(str).tolist() if s.strip()]
                        st.markdown(
                            f'<div class="faca-status-info">ℹ️ Extracted {len(smiles_list)} valid molecules.</div>',
                            unsafe_allow_html=True)
                    else:
                        st.error("❌ Error: The uploaded CSV file must contain a column named 'smiles' or 'SMILES'!")
                except Exception as e:
                    st.error(f"Error reading file: {e}")

        st.markdown("<br>", unsafe_allow_html=True)
        btn_L, btn_C, btn_R = st.columns([5, 3, 5])
        with btn_C:
            start_predict = st.button("Run Prediction", type="primary", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="faca-prompt-result">📊 Prediction Results Overview</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        dummy_results = {
            "SMILES": ["..."] * (min(5, len(smiles_list)) if smiles_list else 1),
            "Prediction": ["..."] * (min(5, len(smiles_list)) if smiles_list else 1),
            "Confidence": ["..."] * (min(5, len(smiles_list)) if smiles_list else 1)
        }
        result_placeholder = st.empty()
        result_placeholder.dataframe(pd.DataFrame(dummy_results), use_container_width=True)

        st.markdown('<br><div class="faca-prompt-result">🔬 Pharmacophore Visualization</div>', unsafe_allow_html=True)
        st.caption(
            "Displays the attention distribution of the top predicted active molecules. Darker colors indicate higher contribution to activity.")
        st.markdown("<br>", unsafe_allow_html=True)

        heatmap_placeholder = st.empty()
        heatmap_placeholder.info("⏳ Waiting for inference to display pharmacophore heatmaps...")

        # ==========================================
        # 🌟 核心修改区：模型推理逻辑 -> 呼叫 api_client 管家
        # ==========================================
        faca_state_key = "faca_batched_results"
        current_faca_input = (tuple(smiles_list), model_key, float(activity_threshold))
        if start_predict:
            st.session_state.pop(faca_state_key, None)
            if not smiles_list:
                st.warning("Please provide valid SMILES structures first!")
            else:
                progress = st.progress(0.0, text=f"Processed 0/{len(smiles_list)} input records")
                def update_faca_progress(done, total):
                    progress.progress(done / total, text=f"Processed {done}/{total} input records")
                with st.spinner("Processing prediction batches..."):
                    try:
                        response = api_client.call_faca_gin_api(
                            smiles_list, model_key, activity_threshold, on_progress=update_faca_progress
                        )
                        st.session_state[faca_state_key] = {
                            "input": current_faca_input, "response": response, "failure_reason": None,
                        }
                    except api_client.BatchPredictionError as exc:
                        st.session_state[faca_state_key] = {
                            "input": current_faca_input, "response": exc.partial_response,
                            "failure_reason": str(exc),
                        }
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")

        saved_faca = st.session_state.get(faca_state_key)
        if saved_faca is not None and saved_faca["input"] == current_faca_input:
            response = saved_faca["response"]
            results_df = pd.DataFrame(response["results_df"])
            completed = response["completed_input_count"]
            complete = response["complete"]
            if complete:
                st.success(f"Prediction complete: {len(results_df)} predictions from {completed} processed inputs; threshold {response['threshold_used']:.2f}.")
            else:
                st.warning(f"Prediction stopped after {completed}/{len(smiles_list)} input records. Completed results are retained.")
                st.error(saved_faca["failure_reason"])
            if response.get("skipped_input_indices"):
                st.caption(f"{len(response['skipped_input_indices'])} processed inputs were omitted by the backend because they could not be parsed.")
            result_placeholder.dataframe(results_df, use_container_width=True)
            if not results_df.empty:
                export_frame = results_df.copy()
                export_frame.insert(0, "Input_record", response["result_input_indices"])
                st.download_button(
                    "⬇️ Download Prediction Results (CSV)" if complete else "⬇️ Download Completed Results (CSV)",
                    export_frame.to_csv(index=False).encode("utf-8-sig"),
                    file_name="FACA_GIN_Results.csv" if complete else "FACA_GIN_Partial_Results.csv",
                    mime="text/csv", use_container_width=True, key="faca_batch_download",
                )
            if not complete:
                remaining = pd.DataFrame({"SMILES": smiles_list[completed:]})
                st.download_button("⬇️ Download Remaining Inputs (CSV)",
                    remaining.to_csv(index=False).encode("utf-8-sig"), file_name="FACA_GIN_Remaining_Inputs.csv",
                    mime="text/csv", use_container_width=True, key="faca_remaining_download")
            heatmap_imgs = response.get("heatmap_imgs", [])
            if heatmap_imgs:
                with heatmap_placeholder.container():
                    for index, item in enumerate(heatmap_imgs):
                        st.markdown(f"**Top {index + 1} Active Molecule:** `{item['smiles']}`")
                        columns = st.columns(2)
                        for column, field, title, suffix in (
                            (columns[0], "frag_image", "🧩 Macroscopic: Fragment-level Attention Heatmap", "Macro"),
                            (columns[1], "atom_image", "🔬 Microscopic: Atom-level Attention Heatmap", "Micro"),
                        ):
                            with column:
                                png = base64.b64decode(item[field])
                                st.image(png, use_container_width=True)
                                st.markdown(f"<p style='text-align: center; color: gray; font-weight: bold;'>{title}</p>", unsafe_allow_html=True)
                                st.download_button(f"⬇️ Download {suffix} Image (PNG)", png,
                                    file_name=f"Top{index + 1}_{suffix}.png", mime="image/png",
                                    use_container_width=True, key=f"faca_{suffix}_{index}")
                        st.markdown("---")
            else:
                heatmap_placeholder.info("No attention heatmaps are available among the completed results.")

elif page == "Multi-target Prediction":
    # ==========================================
    # 🎨 Local CSS Injection: Reusing styles + 🌟 Tech-Blue 3D Gradient Card
    # ==========================================
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #F8FAFC; }
    
    /* 🌟 Mod: Tech-Green 3D gradient card matching the Home page green theme */
    .moe-hero-card {
        background: linear-gradient(135deg, #E0F2FE 0%, #F8FAFC 50%, #CCFBF1 100%); 
        border-radius: 16px;
        padding: 30px;
        max-width: 900px;  
        margin: 20px auto 40px auto; 
        /* 阴影和边框采用偏蓝的深青色 */
        box-shadow: 0 20px 40px rgba(13, 148, 136, 0.05), 0 1px 3px rgba(0,0,0,0.03); 
        border: 1px solid rgba(13, 148, 136, 0.1); 
    }
    
    
    .moe-hero-title { font-size: 38px; font-weight: 900; color: #1e293b; margin-bottom: 15px; }
    .moe-hero-subtitle { color: #475569; font-size: 16px; line-height: 1.8; }
    
    .hero-pill {
        display: inline-block; background-color: white; padding: 6px 18px; border-radius: 20px;
        font-size: 14px; font-weight: 700; color: #3b82f6; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.02); margin-right: 12px; margin-top: 20px;
        border: 1px solid #bfdbfe;
    }
    
    /* Module Title Backgrounds */
    .moe-prompt-desc { font-size: 15px; font-weight: 700; display: block; padding: 8px 16px; border-radius: 8px; margin-top: 10px; margin-bottom: 15px; background-color: #E6FFFA; color: #0D9488; border-left: 5px solid #0D9488; }
    .moe-prompt-data { font-size: 15px; font-weight: 700; display: block; padding: 8px 16px; border-radius: 8px; margin-top: 10px; margin-bottom: 15px; background-color: #E0F2FE; color: #075985; border-left: 5px solid #0284C7; }
    .moe-prompt-result { font-size: 15px; font-weight: 700; display: block; padding: 8px 16px; border-radius: 8px; margin-top: 10px; margin-bottom: 15px; background-color: #DCFCE7; color: #166534; border-left: 5px solid #16A34A; }
    
    /* Status Message Boxes */
    .moe-status-info { font-size: 15px; color: #0369A1; background-color: #F0F9FF; padding: 12px 16px; border-radius: 8px; border: 1px solid #BAE6FD; margin-bottom: 15px; display: block; }
    .moe-status-success { font-size: 15px; color: #15803D; background-color: #F0FDF4; padding: 12px 16px; border-radius: 8px; border: 1px solid #BBF7D0; margin-bottom: 15px; display: block; }

    /* 🚀 Button Beautification */
    div.stButton > button[kind="primary"], div.stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%) !important; 
        color: white !important; border: none !important; border-radius: 8px !important;
        box-shadow: 0 4px 10px rgba(79, 70, 229, 0.3) !important; transition: all 0.3s ease !important;
        font-size: 16px !important; font-weight: bold !important; padding: 10px 0px !important;
    }
    div.stButton > button[kind="primary"]:hover, div.stButton > button[data-testid="baseButton-primary"]:hover {
        transform: translateY(-2px) !important; box-shadow: 0 6px 15px rgba(79, 70, 229, 0.4) !important; color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ==========================================
    # 📐 Page Layout
    # ==========================================
    spacer_left, center_col, spacer_right = st.columns([1, 8, 1])

    with center_col:
        st.markdown("""
        <div class="moe-hero-card">
            <div class="moe-hero-title">Multi-target Prediction</div>
            <div class="moe-hero-subtitle">
                Based on 1D/2D features and 3D spatial geometric representations extracted by Uni-Mol, a customized gating network is employed to achieve joint prediction of agonists for FXR and its crosstalk receptors (PXR, LXR, PPARG).
            </div>
            <span class="hero-pill">Uni-Mol 3D</span>
            <span class="hero-pill">Crosstalk Analysis</span>
            <span class="hero-pill">Downloadable Results</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="moe-prompt-data">🧬 Input Molecular Data</div>', unsafe_allow_html=True)
        input_method = st.radio(
            label="moe_input_method_hidden", 
            options=["✍️ Input SMILES", "📁 Upload CSV File"], 
            horizontal=True, 
            label_visibility="collapsed"
        )
        
        smiles_list = []
        np_ids = []
        if input_method == "✍️ Input SMILES":
            raw_smiles = st.text_area(
                label="moe_smiles_input_hidden", 
                value="",  
                height=150, 
                placeholder="Paste SMILES strings here, use Enter for multiple lines...\nExample: CN(C)C(=O)c1cccc(-c2ccc(OCc3c(-c4c(Cl)cccc4Cl)noc3C3CC3)cc2)c1", 
                label_visibility="collapsed"
            )
            if raw_smiles:
                smiles_list = [s.strip() for s in raw_smiles.split('\n') if s.strip()]
                np_ids = [""] * len(smiles_list)
        else:
            uploaded_file = st.file_uploader(label="moe_csv_upload_hidden", type=["csv"], label_visibility="collapsed")
            if uploaded_file is not None:
                try:
                    # 字符串读取，避免 np_id 前导零丢失或 NA 被解释成缺失值。
                    df = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
                    smiles_list, np_ids, skipped = prepare_ortho_csv_input(df)
                    st.markdown('<div class="moe-status-success">✅ File uploaded successfully!</div>', unsafe_allow_html=True)
                    with st.expander("Click to view data preview"):
                        st.dataframe(df.head(), use_container_width=True)
                    st.info(f"Extracted {len(smiles_list)} non-empty SMILES rows; skipped {skipped} empty rows.")
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                

        st.markdown("<br>", unsafe_allow_html=True)
        
        btn_L, btn_C, btn_R = st.columns([5, 3, 5])
        with btn_C:
            start_predict = st.button("Run Multi-target Prediction", type="primary", use_container_width=True, key="moe_btn")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="moe-prompt-result">📊 Multi-target Prediction Results</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True) 
        st.caption("Displaying prediction results for FXR, PXR, LXR, and PPARG agonists.")
        st.markdown("<br>", unsafe_allow_html=True) 
        
        matrix_placeholder = st.empty()
        result_state_key = "ortho_batched_results_v1"
        current_input = (smiles_list, np_ids)

        if start_predict:
            st.session_state.pop(result_state_key, None)
            if not smiles_list:
                st.warning("Please provide valid SMILES structures first!")
            else:
                progress = st.progress(0.0, text=f"Processed 0/{len(smiles_list)} input records")
                def update_ortho_progress(done, total):
                    progress.progress(done / total, text=f"Processed {done}/{total} input records")
                with st.spinner("Processing multi-target prediction batches..."):
                    try:
                        response = api_client.call_ortho_moe_api(smiles_list, "ORTHO", on_progress=update_ortho_progress)
                        frame = validate_ortho_results(response, smiles_list, np_ids)
                        st.session_state[result_state_key] = {
                            "input": current_input, "frame": frame, "complete": True,
                            "completed_count": len(smiles_list), "failure_reason": None,
                        }
                    except api_client.BatchPredictionError as exc:
                        count = exc.completed_count
                        frame = (validate_ortho_results(exc.partial_response, smiles_list[:count], np_ids[:count])
                                 if count else pd.DataFrame())
                        st.session_state[result_state_key] = {
                            "input": current_input, "frame": frame, "complete": False,
                            "completed_count": count, "failure_reason": str(exc),
                        }
                    except Exception as exc:
                        st.error(f"Multi-target prediction failed: {exc}")

        saved_result = st.session_state.get(result_state_key)
        if saved_result is not None and saved_result["input"] == current_input:
            frame = saved_result["frame"]
            if saved_result["complete"]:
                st.success(f"Prediction complete: {len(frame)} results validated.")
            else:
                st.warning(f"Prediction stopped after {saved_result['completed_count']}/{len(smiles_list)} input records. Completed results are retained.")
                st.error(saved_result["failure_reason"])
            matrix_placeholder.dataframe(frame, use_container_width=True)
            if not frame.empty:
                st.download_button(
                    "⬇️ Download Multi-target Results (CSV)" if saved_result["complete"] else "⬇️ Download Completed Results (CSV)",
                    frame.to_csv(index=False).encode("utf-8-sig"),
                    file_name="Ortho_MoE_MultiTarget_Results.csv" if saved_result["complete"] else "Ortho_MoE_Partial_Results.csv",
                    mime="text/csv", use_container_width=True, key="moe_results_download",
                )
            if not saved_result["complete"]:
                count = saved_result["completed_count"]
                remaining = pd.DataFrame({"np_id": np_ids[count:], "SMILES": smiles_list[count:]})
                st.download_button("⬇️ Download Remaining Inputs (CSV)",
                    remaining.to_csv(index=False).encode("utf-8-sig"), file_name="Ortho_MoE_Remaining_Inputs.csv",
                    mime="text/csv", use_container_width=True, key="moe_remaining_download")
        else:
            matrix_placeholder.info("Submit molecules to generate prediction results.")


elif page == "Data Resources":
    render_data_resources_page(APP_DIR)


elif page == "Help":
    # ==========================================
    # 🎨 局部 CSS：Help & Support 风格
    # ==========================================
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #F8FAFC; }

    .help-header {
        position: relative; text-align: center; padding: 32px 24px;
        margin: 20px auto 30px; background: #ffffff;
        border: 1px solid #dbe4f0; border-radius: 18px;
        box-shadow: 0 8px 24px rgba(79,70,229,0.05);
    }
    .help-header::before {
        content: ""; position: absolute; top: 0; left: 50%; transform: translateX(-50%);
        width: 110px; height: 3px; border-radius: 0 0 4px 4px;
        background: linear-gradient(90deg, #6366f1, #38bdf8);
    }
    .help-title {
        display: flex; align-items: center; justify-content: center; gap: 12px;
        font-size: 36px; font-weight: 900; color: #1e293b; margin-bottom: 12px;
    }
    .help-title-icon {
        display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0;
        width: 48px; height: 48px; border-radius: 14px;
        background: #eef2ff; border: 1px solid #e0e7ff; font-size: 27px;
    }
    .help-desc { font-size: 16px; line-height: 1.8; color: #64748b; max-width: 600px; margin: 0 auto; }

    .help-section-title {
        font-size: 15px; font-weight: 700; padding: 10px 16px; border-radius: 9px;
        margin: 10px 0 18px; border: 1px solid #dbeafe; border-left: 4px solid #3b82f6;
        background: #eff6ff; color: #1e40af;
    }
    .help-section-quick { background: #eef2ff; color: #4338ca; border-color: #e0e7ff; border-left-color: #6366f1; }

    .help-quick-grid {
        display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 16px; margin: 0 0 32px;
    }
    .help-quick-step {
        --step-color: #4f46e5; --step-soft: #eef2ff; --step-border: #e0e7ff;
        position: relative; overflow: hidden; min-width: 0;
        background: linear-gradient(160deg, var(--step-soft) 0%, #ffffff 58%);
        border: 1px solid var(--step-border); border-radius: 14px;
        padding: 22px 20px; box-shadow: 0 4px 14px rgba(15,23,42,0.035);
    }
    .help-quick-step::before {
        content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: var(--step-color);
    }
    .help-quick-step:nth-child(2) { --step-color: #0284c7; --step-soft: #f0f9ff; --step-border: #bae6fd; }
    .help-quick-step:nth-child(3) { --step-color: #0d9488; --step-soft: #f0fdfa; --step-border: #ccfbf1; }
    .help-quick-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; }
    .help-quick-icon {
        display: inline-flex; align-items: center; justify-content: center;
        width: 38px; height: 38px; border-radius: 11px;
        color: var(--step-color); background: #ffffff; border: 1px solid var(--step-border);
    }
    .help-quick-icon svg { display: block; width: 21px; height: 21px; }
    .help-quick-number {
        display: inline-flex; align-items: center; justify-content: center;
        width: 38px; height: 38px; border-radius: 11px;
        background: var(--step-color); color: #ffffff; font-size: 16px; font-weight: 800;
        letter-spacing: 0.5px;
    }
    .help-quick-title { font-size: 16px; font-weight: 700; color: #1e293b; margin-bottom: 10px; }
    .help-quick-text { font-size: 14px; line-height: 1.8; color: #475569; }
    @media (max-width: 700px) {
        .help-header { padding: 26px 18px; }
        .help-title { font-size: 27px; gap: 10px; }
        .help-title-icon { width: 40px; height: 40px; font-size: 23px; }
        .help-quick-grid { grid-template-columns: 1fr; gap: 12px; }
        .help-quick-top { margin-bottom: 14px; }
    }

    .contact-card {
        background: white; border-radius: 16px; padding: 30px; text-align: center;
        border: 1px solid #e2e8f0; box-shadow: 0 10px 25px rgba(0,0,0,0.03); margin-top: 40px;
    }
    .contact-icon { font-size: 40px; margin-bottom: 15px; }
    .contact-title { font-size: 20px; font-weight: 700; color: #0f172a; margin-bottom: 10px; }
    .contact-info { font-size: 15px; color: #3b82f6; font-weight: 600; margin-bottom: 5px; }

    /* 美化 st.expander */
    [data-testid="stExpander"] { background: white !important; border-radius: 10px !important; border: 1px solid #e2e8f0 !important; box-shadow: 0 2px 5px rgba(0,0,0,0.01) !important; margin-bottom: 10px !important; }
    [data-testid="stExpander"] summary p { font-size: 16px !important; font-weight: 600 !important; color: #1e293b !important; }
    </style>
    """, unsafe_allow_html=True)

    spacer_left, center_col, spacer_right = st.columns([2, 6, 2])
    with center_col:
        st.markdown("""
        <div class="help-header">
            <div class="help-title" role="heading" aria-level="1"><span class="help-title-icon" aria-hidden="true">📝</span><span>Help Center & Documentation</span></div>
            <div class="help-desc">Find answers to common questions about data formatting, model execution, and platform usage.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="help-section-title help-section-quick" role="heading" aria-level="2">Quick Start</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="help-quick-grid">
            <div class="help-quick-step">
                <div class="help-quick-top">
                    <span class="help-quick-number">01</span>
                    <span class="help-quick-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16M4 17h16"/><circle cx="9" cy="7" r="3" fill="white"/><circle cx="15" cy="17" r="3" fill="white"/></svg></span>
                </div>
                <div class="help-quick-title">Choose a model</div>
                <div class="help-quick-text">Open FXR Prediction for agonist or antagonist prediction, or Multi-target Prediction for joint prediction across four receptors.</div>
            </div>
            <div class="help-quick-step">
                <div class="help-quick-top">
                    <span class="help-quick-number">02</span>
                    <span class="help-quick-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6M12 17v-5m-3 3 3-3 3 3"/></svg></span>
                </div>
                <div class="help-quick-title">Submit molecules</div>
                <div class="help-quick-text">Enter one SMILES string per line or upload a CSV file containing a smiles column, then run the prediction.</div>
            </div>
            <div class="help-quick-step">
                <div class="help-quick-top">
                    <span class="help-quick-number">03</span>
                    <span class="help-quick-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12m-4-4 4 4 4-4M5 15v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4"/></svg></span>
                </div>
                <div class="help-quick-title">Review and download</div>
                <div class="help-quick-text">Review the prediction results and download the CSV file. Visit Data Resources to query and download benchmark datasets.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="help-section-title" role="heading" aria-level="2">Frequently Asked Questions (FAQ)</div>', unsafe_allow_html=True)

        with st.expander("📌 Q1: What is the required format for inputting SMILES?"):
            st.write("""
            When using the **Input SMILES** option, please ensure that:
            * Each SMILES string is on a **new line**.
            * Do not include headers, IDs, or extra punctuation.
            * Invalid SMILES will be automatically skipped or flagged with an error by the RDKit backend.
            """)

        with st.expander("📌 Q2: How should I format the CSV file for batch upload?"):
            st.write("""
            The platform requires a `.csv` file containing at least one column specifically for SMILES data.
            * The column header must be named exactly **`smiles`** or **`SMILES`** (case-insensitive).
            * Other columns (like IDs or experimental values) are allowed and will simply be ignored during prediction.
            """)

        with st.expander("📌 Q3: The webpage shows an API connection error."):
            st.write("""
            This occurs if the Hugging Face Gradio backend Space is either asleep (due to inactivity) or updating. 
            **Solution:** Please wait 1-2 minutes and refresh the page, or click the prediction button again to wake up the server.
            """)

        with st.expander("📌 Q4: How do I search and download benchmark data?"):
            st.write("""
            Open **Data Resources** and choose FXR Agonist, FXR Antagonist or Multi-task.
            Use **Search benchmark** and **Search in** to search by SMILES text, NURA ID or ChEMBL ID.
            Click **Search** (or press Enter in the search field) to apply the query. Clear the text and search again to show all records.
            SMILES text search is case-sensitive and does not perform chemical substructure matching.
            Source identifiers are shown in the preview but excluded from all CSV downloads.
            Binary datasets support complete or search-result downloads with **smiles** and **label** columns.
            """)

        # 联系方式卡片
        st.markdown("""
        <div class="contact-card">
            <div class="contact-icon">✉️</div>
            <div class="contact-title">Need Further Assistance?</div>
            <div style="color: #64748b; font-size: 14px; margin-bottom: 20px;">If you encounter persistent bugs or wish to collaborate, feel free to reach out.</div>
            <div class="contact-info">📧 Email: 202419131063t@cqu.edu.cn</div>
            <div class="contact-info">💻 GitHub: github.com/bzy9/FXRPred</div>
            <div style="color: #94a3b8; font-size: 13px; margin-top: 15px;">© 2026 FXRPred Platform Development Team</div>
        </div>
        """, unsafe_allow_html=True)
