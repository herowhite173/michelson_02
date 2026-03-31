import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import gridspec
import qrcode
from PIL import Image
import io

# ======================== 全局配置（兼容中文+云端） ========================
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.switch_backend('Agg')


# ======================== 核心计算函数 ========================
def calculate_interference(k, h, wavelength_option, is_mobile=False):
    wavelength_map = {
        "红光 (650 nm)": 650e-9,
        "绿光 (532 nm)": 532e-9,
        "蓝光 (473 nm)": 473e-9,
        "黄光 (589.3 nm)": 589.3e-9
    }
    lamd = wavelength_map.get(wavelength_option, 650e-9)

    N = 128 if is_mobile else 256
    hi = 400e-3
    ym = 250e-3
    h1 = h * 1e-9

    x = np.linspace(-ym, ym, N)
    y = np.linspace(-ym, ym, N)
    X, Y = np.meshgrid(x, y)

    r2 = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan(r2 / hi)
    di = lamd + k * lamd / 2 + h1
    delta = 2 * di * np.cos(theta)
    phi = 2 * np.pi * delta / lamd
    I = 4 * 10 * np.cos(phi / 2) ** 2
    I = I / np.max(I) if np.max(I) != 0 else I

    cmap_dict = {650e-9: 'Reds', 532e-9: 'Greens', 473e-9: 'Blues', 589.3e-9: 'YlOrBr'}
    cmap = cmap_dict.get(lamd, 'viridis')

    if is_mobile:
        fig_size = (10, 5)
        dpi = 96
    else:
        fig_size = (16, 8)
        dpi = 120

    fig = plt.figure(figsize=fig_size, dpi=dpi)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])

    ax1 = fig.add_subplot(gs[0])
    ax1.set_aspect('equal')
    h2 = h * 1e-2
    if h > 300:
        h2 = 300e-2

    ax1.plot([-6, 6], [18, 18], '-', color='g', lw=2, label='M2（固定镜）')
    ax1.plot([20, 20], [-8, 4], '-', color='g', lw=2, label='M1（移动镜）')
    ax1.plot([-6, 6], [22 + h2, 22 + h2], '--', color='r', lw=2, label="M2'（虚像）")
    ax1.text(10, 15, "M1", fontsize=12, color='g')
    ax1.text(10, 21 + h2, "M2'", fontsize=12, color='r')
    ax1.text(18, 5, 'M2', fontsize=12, color='g')
    ax1.plot([-4, 4], [-6, 2], '-', color='k', lw=2, label='分束镜')
    ax1.plot([4, 12], [-6, 2], '-', color='k', lw=2)
    ax1.plot([0, 0], [-22, 18], '-', color='r', lw=1, label='入射光路')
    ax1.plot([-20, 20], [-2, -2], '-', color='r', lw=0.7, label='反射光路')
    ax1.set_title('迈克尔逊干涉原理图', fontsize=18, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax1.set_ylim(-28, 28)
    ax1.set_xlim(-28, 28)
    ax1.set_facecolor('lightgray')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[1])
    extent = [-10, 10, -10, 10]
    im = ax2.imshow(I, cmap=cmap, extent=extent, origin='lower')
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.set_title(f"迈克尔逊等倾干涉条纹（{wavelength_option}）", fontsize=18, fontweight='bold', pad=10)
    ax2.set_xlabel("x (mm)", fontsize=11)
    ax2.set_ylabel("y (mm)", fontsize=11)

    ax2.grid(True, alpha=0.3, linestyle='--')
    cbar = plt.colorbar(im, ax=ax2, label='相对光强', shrink=0.85)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout(pad=2.0)
    return fig


# ======================== 二维码生成函数 ========================
def generate_qr_code(text):
    if not text:
        return None
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=6,
        border=2
    )
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ======================== 主界面 ========================
def main():
    # ========== 关键修改：添加强制横屏的meta标签 ==========
    st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1.0, orientation=landscape">
    """, unsafe_allow_html=True)

    st.set_page_config(
        page_title="迈克尔逊干涉实验仿真",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # 自动判断设备
    try:
        width = st.get_script_run_ctx().device_width
        is_mobile = width < 768
    except:
        is_mobile = False

    # 自动获取当前页面 URL（核心修复）
    try:
        current_url = st.experimental_get_query_params().get("url", [st.server.get_current_url()])[0]
    except:
        current_url = "http://localhost:8501"

    # ====================== 固定布局CSS ======================
    st.markdown("""
    <style>
    .block-container { padding: 1rem 1.5rem !important; max-width: 1400px !important; }
    .main-header { text-align:center; padding:0.8rem; background:linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                  color:white; border-radius:8px; margin-bottom:1.2rem; }
    .stMarkdown h3 {
     font-size: 1rem !important;
     margin-top:0.2rem !important;
     margin-bottom:0.6rem !important;
      }
    .stNumberInput, .stSelectbox, .stButton { margin-bottom:0.2rem !important; }
    .experiment-principle {
        margin-top: -10px !important;
        padding-top: 0 !important;
    }
    @media (max-width: 768px) {
        [data-testid="column"] { width:100% !important; flex:0 0 100% !important; }
        .block-container { padding:0.8rem 1rem !important; }
    }
    </style>
    """, unsafe_allow_html=True)

    # 标题
    st.markdown("""
    <div class="main-header">
        <h1>🔬 迈克尔逊干涉实验仿真</h1>
        <p>云端版 | 实时交互 | 多端适配</p>
    </div>
    """, unsafe_allow_html=True)

    # ========== 布局 ==========
    if is_mobile:
        col_params = st.container()
        col_plot = st.container()
    else:
        col_plot, col_params = st.columns([3, 1], gap="medium")

    # ========== 右侧/下侧：参数和二维码 ==========
    with col_params:
        st.markdown("### ⚙️ 参数调节")
        if 'k' not in st.session_state:
            st.session_state['k'] = 50
            st.session_state['h'] = 100
            st.session_state['wavelength'] = "红光 (650 nm)"

        c1, c2 = st.columns(2)
        with c1:
            if st.button("📌 经典红光参数", use_container_width=True):
                st.session_state.update({'k': 50, 'h': 100, 'wavelength': "红光 (650 nm)"})
        with c2:
            if st.button("🔄 重置参数", use_container_width=True):
                st.session_state.update({'k': 50, 'h': 100, 'wavelength': "红光 (650 nm)"})

        k = st.number_input("**干涉级次 K**", 1, 200, st.session_state.k, 10)
        h = st.number_input("**间距 h (nm)**", 100, 10000, st.session_state.h, 100)
        wavelength = st.selectbox(
            "**入射光波长**",
            ["红光 (650 nm)", "绿光 (532 nm)", "蓝光 (473 nm)", "黄光 (589.3 nm)"],
            index=["红光 (650 nm)", "绿光 (532 nm)", "蓝光 (473 nm)", "黄光 (589.3 nm)"].index(
                st.session_state.wavelength)
        )
        st.session_state.update({'k': k, 'h': h, 'wavelength': wavelength})

        st.markdown("### 📱 扫码访问")
        qr_img = generate_qr_code(current_url)
        if qr_img:
            st.image(qr_img, caption="扫码在手机打开", width=200)
        else:
            st.info("正在生成二维码...")

    # ========== 左侧/上侧：图像和实验原理 ==========
    with col_plot:
        try:
            with st.spinner('🔄 正在计算干涉图样...'):
                fig = calculate_interference(k, h, wavelength, is_mobile)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        except Exception as e:
            st.error(f"❌ 计算出错：{str(e)[:80]}")

        st.markdown("""
        <div class="experiment-principle">
            <h3>📚 实验原理</h3>
            <div style="background:lightyellow; border:2px solid gray; padding:0.8rem; border-radius:5px;">
             实验原理：迈克逊干涉实验类型为等倾干涉，即具有相同入射角度的光线所形成的干涉图样为一个同心圆<br>
            明纹条件: 2d×cosθ=Kλ，（K=1,2,3...）<br>
            暗纹条件: 2d×cosθ=(2K+1)λ×1/2，（K=1,2,3...）<br>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#6c757d; padding:1rem;'>
    基于 Streamlit Cloud 部署 | 波动光学实验仿真<br>© 2026 全国大学生物理竞赛参赛作品
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()