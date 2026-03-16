import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import gridspec
import io
import warnings

# 忽略所有警告
warnings.filterwarnings('ignore')

# ======================== 全局配置（中文显示正常） ========================
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.switch_backend('Agg')

# ======================== 核心计算函数（对齐EXE条纹 + 缓存加速） ========================
@st.cache_data(show_spinner=False, ttl=3600)
def calculate_interference(k, h, wavelength_option, is_mobile=False):
    # 波长映射
    wavelength_map = {
        "红光 (650 nm)": 650e-9,
        "绿光 (532 nm)": 532e-9,
        "蓝光 (473 nm)": 473e-9,
        "黄光 (589.3 nm)": 589.3e-9
    }
    lamd = wavelength_map.get(wavelength_option, 650e-9)

    # 分辨率：和EXE一致，保证清晰
    N = 512 if is_mobile else 800
    hi = 400e-3
    ym = 250e-3
    h1 = h * 1e-9  # nm → m

    # 生成坐标网格
    x = np.linspace(-ym, ym, N)
    y = np.linspace(-ym, ym, N)
    X, Y = np.meshgrid(x, y)

    # ✅ 物理计算：完全对齐EXE，条纹清晰有明暗对比
    r2 = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan(r2 / hi)
    delta = 2 * h1 * np.cos(theta)  # 光程差 = 2h cosθ
    phi = 2 * np.pi * delta / lamd
    I = np.cos(phi / 2) ** 2  # 不做额外归一化，保留明暗对比

    # 配色：和图一一致（黄光用YlOrBr，红光Reds，蓝光Blues，绿光Greens）
    cmap_dict = {
        650e-9: 'Reds',
        532e-9: 'Greens',
        473e-9: 'Blues',
        589.3e-9: 'YlOrBr'
    }
    cmap = cmap_dict.get(lamd, 'YlOrBr')

    # 绘图尺寸
    fig_size = (10, 5) if is_mobile else (12, 6)
    fig = plt.figure(figsize=fig_size, dpi=90)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])

    # -------------------------- 原理图（仅标题用英文） --------------------------
    ax1 = fig.add_subplot(gs[0])
    ax1.set_aspect('equal')
    h2 = h * 1e-2
    if h > 300:
        h2 = 300e-2

    # 绘制光学元件
    ax1.plot([-6, 6], [18, 18], '-', color='g', linewidth=2, label='M2')
    ax1.plot([20, 20], [-8, 4], '-', color='g', linewidth=2, label='M1')
    ax1.plot([-6, 6], [22+h2, 22+h2], '--', color='g', linewidth=2, label="M2'")

    # 文本标注
    ax1.text(10, 15, "M1", fontsize=12, color='g')
    ax1.text(10, 21+h2, "M2'", fontsize=12, color='g')
    ax1.text(18, 5, 'M2', fontsize=12, color='g')

    # 光路绘制
    ax1.plot([-4, 4], [-6, 2], '-', color='k', linewidth=2)
    ax1.plot([4, 12], [-6, 2], '-', color='k', linewidth=2)
    ax1.plot([0, 0], [-22, 18], '-', color='r', linewidth=1)
    ax1.plot([-20, 20], [-2, -2], '-', color='r', linewidth=0.7)

    # ✅ 仅标题用英文，图例正常显示
    ax1.set_title('Schematic Diagram', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax1.set_ylim(-28, 28)
    ax1.set_xlim(-28, 28)
    ax1.set_facecolor('lightgray')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # -------------------------- 干涉图样（仅标题用英文） --------------------------
    ax2 = fig.add_subplot(gs[1])
    extent = [-10, 10, -10, 10]
    im = ax2.imshow(I, cmap=cmap, extent=extent, origin='lower', vmin=0, vmax=1)

    # 刻度与标签
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.set_title("Michelson Interference", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("x (mm)", fontsize=11)
    ax2.set_ylabel("y (mm)", fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 固定坐标轴，避免跳动
    ax2.set_xlim(-10, 10)
    ax2.set_ylim(-10, 10)

    # 颜色条（可选，图一没有就注释掉）
    # cbar = plt.colorbar(im, ax=ax2, label='相对光强', shrink=0.85)
    # cbar.ax.tick_params(labelsize=9)

    plt.tight_layout(pad=2.0)
    return fig

# ======================== 主界面（全部保留中文） ========================
def main():
    st.set_page_config(
        page_title="迈克尔逊干涉实验仿真",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # 移动端检测
    try:
        is_mobile = st.query_params.get("mobile", "") == "true"
    except:
        is_mobile = False

    # 自定义CSS
    st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stNumberInput label, .stSelectbox label {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }
    .param-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .principle-box {
        font-size: 12pt;
        color: darkblue;
        background-color: lightyellow;
        border: 2px solid gray;
        border-radius: 5px;
        padding: 10px;
        margin-top: 1rem;
        line-height: 1.5;
    }
    </style>
    """, unsafe_allow_html=True)

    # 页面标题（中文）
    st.markdown("""
    <div class="main-header">
        <h1>🔬 迈克尔逊干涉实验仿真</h1>
        <p>云端版 | 实时交互 | 多端适配</p>
    </div>
    """, unsafe_allow_html=True)

    # 布局
    if is_mobile:
        col1, col2 = st.container(), st.container()
    else:
        col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📱 扫码访问")
        st.info("""
        💡 请使用浏览器地址栏中的 URL 生成二维码，或直接分享该链接。
        手机扫码或点击链接即可访问。
        """)

        st.markdown("### ⚙️ 参数调节")
        with st.container():
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("📌 经典红光参数", use_container_width=True):
                    st.session_state.update({'k': 50, 'h': 20.0, 'wavelength': "红光 (650 nm)"})
            with col_btn2:
                if st.button("🔄 重置参数", use_container_width=True):
                    st.session_state.update({'k': 50, 'h': 20.0, 'wavelength': "红光 (650 nm)"})

            # 初始化会话状态
            if 'k' not in st.session_state:
                st.session_state['k'] = 50
                st.session_state['h'] = 20.0
                st.session_state['wavelength'] = "红光 (650 nm)"

            # 点击式参数输入（中文标签，不卡顿）
            k = st.number_input(
                "**干涉级次: K**",
                min_value=1,
                max_value=200,
                value=st.session_state['k'],
                step=1,
                format="%d",
                help="干涉条纹级次，正整数 | 取值范围：1-200"
            )
            h = st.number_input(
                "**间距: h/nm**",
                min_value=10.0,
                max_value=2000.0,
                value=float(st.session_state['h']),
                step=10.0,
                format="%.2f",
                help="M1与M2'的空气膜厚度（nm） | 取值范围：10-2000nm"
            )
            wavelength = st.selectbox(
                "**波长:**",
                options=["红光 (650 nm)", "绿光 (532 nm)", "蓝光 (473 nm)", "黄光 (589.3 nm)"],
                index=["红光 (650 nm)", "绿光 (532 nm)", "蓝光 (473 nm)", "黄光 (589.3 nm)"].index(st.session_state['wavelength'])
            )

            # 更新会话状态
            st.session_state.update({'k': k, 'h': h, 'wavelength': wavelength})

        st.markdown("### 📊 当前参数")
        st.markdown(f"""
        <div class="param-card">
            • 干涉级次 K = <strong>{k}</strong><br>
            • 间距 h = <strong>{h}</strong> nm<br>
            • 波长 = <strong>{wavelength}</strong>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📚 实验原理")
        st.markdown("""
        <div class="principle-box">
        实验原理：迈克尔逊干涉实验类型为等倾干涉，即具有相同入射角度的光线所形成的干涉图样为一个同心圆
        <br>明纹条件:  2d×cosθ=Kλ，（K=1,2,3...）
        <br>暗纹条件:  2d×cosθ=(2K+1)λ×1/2，（K=1,2,3...）
        <br>可控参数：干涉级次K，间距h，波长λ
        </div>
        """, unsafe_allow_html=True)

    with col2:
        try:
            with st.spinner('🔄 正在计算干涉图样...'):
                fig = calculate_interference(k, h, wavelength, is_mobile)
                st.pyplot(fig, width='stretch')
                plt.close(fig)
            st.success(f"✅ 计算完成 | K={k} | h={h}nm | {wavelength}")
        except Exception as e:
            st.error(f"❌ 计算出错：{str(e)[:80]}...")

    # 页脚（中文）
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d; padding: 1rem; font-size: 0.85rem;'>
        <p>基于 Streamlit Cloud 部署 | 波动光学实验仿真</p>
        <p style='font-size: 0.75rem;'>© 2026 全国大学生物理竞赛参赛作品</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()