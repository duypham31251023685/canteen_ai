import streamlit as st
import cv2
import tempfile
import pandas as pd
import os
import numpy as np
import time
from datetime import datetime
from crop import main
from predict import run_prediction_pipeline

DISCOUNT_CODES = {
    "MAMMAM10": 0.10,
    "UEH20": 0.20,
    "NHOM7": 0.15
}

FOOD_INFO = {
    "Cơm trắng": {"calories": 250},
    "Sườn nướng": {"calories": 350},
    "Canh rau": {"calories": 40},
    "Canh chua": {"calories": 80},
    "Rau xào": {"calories": 120},
    "Thịt kho": {"calories": 280},
    "Thịt kho trứng": {"calories": 350},
    "Trứng chiên": {"calories": 180},
    "Đậu hũ sốt cà": {"calories": 150},
    "Cá hú kho": {"calories": 300}
}

def save_bill_history(bill):
    filename = "history.csv"
    row = {
        "time": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "total": bill["total"]
    }
    df_new = pd.DataFrame([row])
    if os.path.exists(filename):
        df_old = pd.read_csv(filename)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(filename, index=False)

def generate_vietqr(amount):
    bank_id = "970422"
    account_no = "0798801616"
    account_name = "PHAM NGUYEN BAO CHAU"
    qr_url = (
        f"https://img.vietqr.io/image/"
        f"{bank_id}-{account_no}-compact2.png"
        f"?amount={amount}"
        f"&addInfo=ThanhToanKhayCom"
        f"&accountName={account_name}"
    )
    return qr_url

st.set_page_config(
    page_title="Canteen AI",
    page_icon="🍽️",
    layout="wide"
)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Reset & Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #2C2B5E;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── App background ── */
.stApp {
    background: #FFFFFF;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #434291 0%, #8083D6 100%) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: rgba(255,255,255,0.85) !important;
    font-weight: 500;
    padding: 6px 0;
    transition: color 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: #9CD254 !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: #FFFFFF !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #9CD254 !important;
    color: #2C2B5E !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(156, 210, 84, 0.35) !important;
}
.stButton > button:hover {
    background: #86bb3f !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(156, 210, 84, 0.45) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #F4F5FB;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: #8083D6 !important;
    font-weight: 600 !important;
    background: transparent !important;
    border: none !important;
    padding: 8px 24px !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: #434291 !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 8px rgba(67,66,145,0.3) !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #434291, #9CD254) !important;
    border-radius: 8px;
}
.stProgress > div > div {
    background: #F4F5FB !important;
    border-radius: 8px;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #F4F5FB !important;
    border: 1.5px solid #8083D6 !important;
    border-radius: 16px !important;
    padding: 20px 24px !important;
    box-shadow: 0 2px 12px rgba(128,131,214,0.12) !important;
}
[data-testid="metric-container"] label {
    color: #8083D6 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #434291 !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #F4F5FB !important;
    border: 2px dashed #8083D6 !important;
    border-radius: 16px !important;
    padding: 16px !important;
}
[data-testid="stFileUploader"] label {
    color: #434291 !important;
    font-weight: 600 !important;
}

/* ── Camera input ── */
[data-testid="stCameraInput"] {
    background: #F4F5FB !important;
    border-radius: 16px !important;
    border: 2px dashed #8083D6 !important;
    overflow: hidden;
}

/* ── Number input ── */
[data-testid="stNumberInput"] > div {
    border: 1.5px solid #8083D6 !important;
    border-radius: 10px !important;
    background: #F4F5FB !important;
}

/* ── Success / Info / Warning ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: none !important;
    font-weight: 500 !important;
}
div[data-baseweb="notification"][data-kind="positive"],
.element-container .stSuccess {
    background: linear-gradient(135deg, #f0fce8, #e4f7cc) !important;
    border-left: 4px solid #9CD254 !important;
    border-radius: 12px !important;
    color: #2C2B5E !important;
}
div[data-baseweb="notification"][data-kind="info"],
.element-container .stInfo {
    background: linear-gradient(135deg, #ededfb, #e3e4f8) !important;
    border-left: 4px solid #8083D6 !important;
    border-radius: 12px !important;
    color: #2C2B5E !important;
}
div[data-baseweb="notification"][data-kind="warning"],
.element-container .stWarning {
    background: linear-gradient(135deg, #fff8e6, #fef3cc) !important;
    border-left: 4px solid #f5c842 !important;
    border-radius: 12px !important;
    color: #2C2B5E !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    height: 1.5px !important;
    background: linear-gradient(90deg, transparent, #8083D6, transparent) !important;
    margin: 24px 0 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1.5px solid #8083D6 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] > div {
    border-top-color: #434291 !important;
}

/* ── Line chart ── */
[data-testid="stVegaLiteChart"] {
    border-radius: 16px !important;
    border: 1.5px solid #8083D6 !important;
    overflow: hidden !important;
    background: #F4F5FB !important;
    padding: 12px !important;
}

/* ── Image captions ── */
[data-testid="stImage"] > div > p {
    color: #8083D6 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-align: center !important;
}

/* ── Headings ── */
h1, h2, h3 { color: #2C2B5E !important; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
if "image" not in st.session_state:
    st.session_state.image = None

if "original_image" not in st.session_state:
    st.session_state.original_image = None

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = None

if "promo_applied" not in st.session_state:
    st.session_state.promo_applied = None

if "promo_discount_rate" not in st.session_state:
    st.session_state.promo_discount_rate = 0.0

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center; padding: 24px 0 16px 0;">
    <div style="font-size:48px; margin-bottom:8px;">🍽️</div>
    <div style="font-size:22px; font-weight:800; color:#FFFFFF; letter-spacing:1px;">CANTEEN AI</div>
    <div style="font-size:12px; color:rgba(255,255,255,0.7); font-weight:500; margin-top:4px;">Smart Food Recognition</div>
</div>
<hr style="border:none; height:1px; background:rgba(255,255,255,0.25); margin:0 0 20px 0;">
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "📋 Điều hướng",
    [
        "🏠 Trang chủ",
        "🍽️ Nhận diện món ăn",
        "📊 Dashboard"
    ]
)

st.sidebar.markdown("""
<div style="position:fixed; bottom:24px; left:0; width:280px; text-align:center; padding: 0 16px;">
    <div style="font-size:11px; color:rgba(255,255,255,0.5); font-weight:500;">
        Powered by Ensemble CNN · VietQR
    </div>
</div>
""", unsafe_allow_html=True)

# ─── PAGE: TRANG CHỦ ──────────────────────────────────────────────────────────
if page == "🏠 Trang chủ":

    st.markdown("""
    <div style="
        padding: 56px 48px;
        border-radius: 28px;
        background: linear-gradient(135deg, #434291 0%, #6B68C8 50%, #8083D6 100%);
        text-align: center;
        margin-bottom: 32px;
        box-shadow: 0 12px 40px rgba(67,66,145,0.3);
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position:absolute; top:-60px; right:-60px;
            width:220px; height:220px; border-radius:50%;
            background:rgba(255,255,255,0.06);
        "></div>
        <div style="
            position:absolute; bottom:-40px; left:-40px;
            width:160px; height:160px; border-radius:50%;
            background:rgba(156,210,84,0.12);
        "></div>
        <div style="position:relative; z-index:1;">
            <div style="font-size:64px; margin-bottom:16px;">🍽️</div>
            <h1 style="
                color: #FFFFFF; font-size:42px; font-weight:800;
                margin:0 0 12px 0; letter-spacing:-0.5px;
            ">CANTEEN AI</h1>
            <p style="
                color:rgba(255,255,255,0.85); font-size:18px;
                font-weight:500; margin:0 0 10px 0;
            ">Hệ thống nhận diện món ăn và tính hóa đơn tự động</p>
            <p style="
                color:rgba(255,255,255,0.65); font-size:14px; margin:0;
            ">Upload hoặc chụp ảnh khay cơm để AI nhận diện món ăn, tính tiền và tạo mã QR thanh toán.</p>
            <div style="
                display:inline-block; margin-top:28px;
                background:#9CD254; color:#2C2B5E;
                padding:10px 32px; border-radius:50px;
                font-weight:700; font-size:14px;
                box-shadow: 0 4px 16px rgba(156,210,84,0.4);
            ">🚀 Sẵn sàng hoạt động</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="
            background:#F4F5FB; border:1.5px solid #8083D6;
            border-radius:20px; padding:28px 20px; text-align:center;
            box-shadow:0 2px 12px rgba(128,131,214,0.12);
        ">
            <div style="font-size:36px; margin-bottom:12px;">🍚</div>
            <div style="font-size:36px; font-weight:800; color:#434291;">10</div>
            <div style="font-size:13px; font-weight:600; color:#8083D6; margin-top:4px; text-transform:uppercase; letter-spacing:0.5px;">Món hỗ trợ</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
            background:#F4F5FB; border:1.5px solid #8083D6;
            border-radius:20px; padding:28px 20px; text-align:center;
            box-shadow:0 2px 12px rgba(128,131,214,0.12);
        ">
            <div style="font-size:36px; margin-bottom:12px;">🤖</div>
            <div style="font-size:28px; font-weight:800; color:#434291;">Ensemble CNN</div>
            <div style="font-size:13px; font-weight:600; color:#8083D6; margin-top:4px; text-transform:uppercase; letter-spacing:0.5px;">AI Model</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="
            background:#F4F5FB; border:1.5px solid #8083D6;
            border-radius:20px; padding:28px 20px; text-align:center;
            box-shadow:0 2px 12px rgba(128,131,214,0.12);
        ">
            <div style="font-size:36px; margin-bottom:12px;">💳</div>
            <div style="font-size:36px; font-weight:800; color:#434291;">VietQR</div>
            <div style="font-size:13px; font-weight:600; color:#8083D6; margin-top:4px; text-transform:uppercase; letter-spacing:0.5px;">Thanh toán</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #ededfb, #e3e4f8);
        border-left: 4px solid #8083D6;
        border-radius: 12px;
        padding: 16px 20px;
        color: #2C2B5E;
        font-weight: 500;
        font-size: 15px;
    ">
        🚀 &nbsp; Chọn mục <strong>Nhận diện món ăn</strong> ở menu bên trái để bắt đầu sử dụng hệ thống.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background:#F4F5FB; border:1.5px solid #8083D6;
        border-radius:20px; padding:28px 32px;
        box-shadow:0 2px 12px rgba(128,131,214,0.10);
    ">
        <div style="font-size:16px; font-weight:700; color:#434291; margin-bottom:20px;">⚡ Hướng dẫn sử dụng nhanh</div>
        <div style="display:flex; flex-direction:column; gap:14px;">
            <div style="display:flex; align-items:center; gap:14px;">
                <div style="
                    min-width:36px; height:36px; border-radius:50%;
                    background:#434291; color:#fff;
                    display:flex; align-items:center; justify-content:center;
                    font-weight:700; font-size:16px;
                ">1</div>
                <div style="color:#2C2B5E; font-size:14px; font-weight:500;">Chụp hoặc upload ảnh khay cơm của bạn</div>
            </div>
            <div style="display:flex; align-items:center; gap:14px;">
                <div style="
                    min-width:36px; height:36px; border-radius:50%;
                    background:#434291; color:#fff;
                    display:flex; align-items:center; justify-content:center;
                    font-weight:700; font-size:16px;
                ">2</div>
                <div style="color:#2C2B5E; font-size:14px; font-weight:500;">AI tự động nhận diện 5 ngăn khay và tên món ăn</div>
            </div>
            <div style="display:flex; align-items:center; gap:14px;">
                <div style="
                    min-width:36px; height:36px; border-radius:50%;
                    background:#434291; color:#fff;
                    display:flex; align-items:center; justify-content:center;
                    font-weight:700; font-size:16px;
                ">3</div>
                <div style="color:#2C2B5E; font-size:14px; font-weight:500;">Xem hóa đơn và quét mã QR để thanh toán ngay</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── PAGE: NHẬN DIỆN MÓN ĂN ──────────────────────────────────────────────────
elif page == "🍽️ Nhận diện món ăn":

    st.markdown("""
    <div style="
        padding: 28px 36px;
        border-radius: 20px;
        background: linear-gradient(135deg, #434291, #8083D6);
        margin-bottom: 28px;
        box-shadow: 0 8px 24px rgba(67,66,145,0.25);
    ">
        <h2 style="color:#FFFFFF; margin:0 0 6px 0; font-size:26px; font-weight:800;">📸 Nhận diện món ăn</h2>
        <p style="color:rgba(255,255,255,0.75); margin:0; font-size:14px; font-weight:500;">
            Upload hoặc chụp ảnh khay cơm để AI phân tích và tính hóa đơn tự động
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        font-size:15px; font-weight:700; color:#434291;
        margin-bottom:12px; letter-spacing:0.2px;
    ">📂 Chọn nguồn ảnh</div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📁 Upload ảnh", "📷 Chụp từ Camera"])

    uploaded_file = None
    camera_file = None

    with tab1:
        uploaded_file = st.file_uploader(
            "Upload ảnh khay cơm",
            type=["jpg", "jpeg", "png"]
        )

    with tab2:
        camera_file = st.camera_input("Chụp ảnh khay cơm")

    image_source = None

    if uploaded_file is not None:
        image_source = uploaded_file
    elif camera_file is not None:
        image_source = camera_file

    if image_source is not None:
        file_bytes = image_source.read()

        image = cv2.imdecode(
            np.frombuffer(file_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        if st.session_state.original_image is None:
            st.session_state.original_image = image.copy()

        if st.session_state.image is None:
            st.session_state.image = image.copy()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:15px; font-weight:700; color:#434291; margin-bottom:12px;">🖼️ Xem trước & Chỉnh sửa ảnh</div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if st.button("↩️ Xoay trái"):
                st.session_state.image = cv2.rotate(
                    st.session_state.image,
                    cv2.ROTATE_90_COUNTERCLOCKWISE
                )

        with col2:
            if st.button("↪️ Xoay phải"):
                st.session_state.image = cv2.rotate(
                    st.session_state.image,
                    cv2.ROTATE_90_CLOCKWISE
                )

        with col3:
            if st.button("🔄 Lật ngang"):
                st.session_state.image = cv2.flip(
                    st.session_state.image,
                    1
                )

        with col4:
            if st.button("🔃 Lật dọc"):
                st.session_state.image = cv2.flip(
                    st.session_state.image,
                    0
                )

        with col5:
            if st.button("🗑️ Reset"):
                st.session_state.image = (
                    st.session_state.original_image.copy()
                )

        st.markdown("""
        <div style="
            border:1.5px solid #8083D6; border-radius:16px;
            overflow:hidden; margin:8px 0 20px 0;
            box-shadow:0 4px 16px rgba(128,131,214,0.15);
        ">
        """, unsafe_allow_html=True)
        st.image(
            cv2.cvtColor(
                st.session_state.image,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🚀 Bắt đầu nhận diện"):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            ) as tmp:
                cv2.imwrite(tmp.name, st.session_state.image)
                image_path = tmp.name

            progress_bar = st.progress(0)
            status = st.empty()

            status.info("📷 Đang chuẩn bị ảnh...")
            progress_bar.progress(10)

            with st.spinner("📷 Đang xử lý ảnh khay cơm..."):

                time.sleep(0.5)

                status.info("✂️ Đang cắt 5 ngăn khay...")
                progress_bar.progress(40)

                rotated_img, compartments = main(
                    use_demo=False,
                    image_path=image_path,
                    run_calibration=False
                )

                time.sleep(0.5)

                status.info("🤖 Ensemble CNN đang nhận diện món ăn...")
                progress_bar.progress(75)

                bill = run_prediction_pipeline(compartments)

                time.sleep(0.5)

                status.info("💰 Đang tính hóa đơn...")
                progress_bar.progress(95)

                save_bill_history(bill)

            progress_bar.progress(100)
            status.success("✅ Nhận diện hoàn tất!")

            st.balloons()

            time.sleep(1)

            progress_bar.empty()
            status.empty()

            st.markdown("""
            <div style="
                background:linear-gradient(135deg,#f0fce8,#e4f7cc);
                border-left:4px solid #9CD254;
                border-radius:12px; padding:14px 20px;
                color:#2C2B5E; font-weight:600; font-size:15px;
                margin-bottom:20px;
            ">🎉 Hoàn tất xử lý! AI đã nhận diện xong khay cơm của bạn.</div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="font-size:15px; font-weight:700; color:#434291; margin-bottom:10px;">📐 Ảnh sau khi căn chỉnh</div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style="border:1.5px solid #8083D6; border-radius:16px; overflow:hidden; box-shadow:0 4px 16px rgba(128,131,214,0.15);">
            """, unsafe_allow_html=True)
            st.image(
                cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB),
                caption="Ảnh sau xoay",
                use_container_width=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:15px; font-weight:700; color:#434291; margin-bottom:12px;">🖼️ 5 ngăn sau khi crop</div>
            """, unsafe_allow_html=True)

            cols = st.columns(5)

            slot_names = {
                "soup": "Canh",
                "rice": "Cơm",
                "comp1": "Món 1",
                "comp2": "Món 2",
                "comp3": "Món 3"
            }

            for i, (slot, img) in enumerate(compartments.items()):
                with cols[i]:
                    st.markdown(f"""
                    <div style="
                        border:1.5px solid #8083D6; border-radius:12px;
                        overflow:hidden; box-shadow:0 2px 8px rgba(128,131,214,0.12);
                    ">
                    """, unsafe_allow_html=True)
                    st.image(
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                        caption=slot_names.get(slot, slot),
                        use_container_width=True
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:15px; font-weight:700; color:#434291; margin-bottom:14px;">🧾 Kết quả nhận diện</div>
            """, unsafe_allow_html=True)

            extra_egg_cost = 0

            for item in bill["items"]:
                st.markdown(f"""
                <div style="
                    background:#F4F5FB; border:1.5px solid #8083D6;
                    border-radius:16px; padding:20px 24px; margin-bottom:14px;
                    box-shadow:0 2px 10px rgba(128,131,214,0.1);
                ">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
                        <div>
                            <div style="font-size:16px; font-weight:700; color:#434291; margin-bottom:6px;">{item['label']}</div>
                            <div style="font-size:20px; font-weight:800; color:#2C2B5E; margin-bottom:6px;">🍽️ {item['food']}</div>
                            <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                                <span style="
                                    background:#434291; color:#fff;
                                    padding:3px 12px; border-radius:50px;
                                    font-size:12px; font-weight:600;
                                ">Độ tin cậy: {item['confidence']:.1%}</span>
                            </div>
                        </div>
                        <div style="
                            background:linear-gradient(135deg,#434291,#8083D6);
                            color:#fff; padding:12px 24px; border-radius:14px;
                            text-align:center; min-width:110px;
                            box-shadow:0 4px 12px rgba(67,66,145,0.25);
                        ">
                            <div style="font-size:11px; opacity:0.8; margin-bottom:2px; font-weight:600;">GIÁ</div>
                            <div style="font-size:22px; font-weight:800;">{item['price']:,}đ</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if item["note"]:
                    st.markdown(f"""
                    <div style="
                        background:#fff8e6; border-left:4px solid #f5c842;
                        border-radius:10px; padding:10px 16px; margin-bottom:10px;
                        color:#2C2B5E; font-size:14px; font-weight:500;
                    ">⚠️ {item['note']}</div>
                    """, unsafe_allow_html=True)

                if item["food"] == "Thịt kho trứng":
                    extra_egg = st.number_input(
                        "🥚 Thêm trứng",
                        min_value=0,
                        max_value=5,
                        value=0,
                        key=f"egg_{item['slot']}"
                    )
                    extra_egg_cost += extra_egg * 6000
                    if extra_egg > 0:
                        st.markdown(f"""
                        <div style="
                            background:linear-gradient(135deg,#ededfb,#e3e4f8);
                            border-left:4px solid #8083D6;
                            border-radius:10px; padding:10px 16px; margin-bottom:10px;
                            color:#2C2B5E; font-size:14px; font-weight:600;
                        ">➕ Phụ thu thêm trứng: {extra_egg * 6000:,}đ</div>
                        """, unsafe_allow_html=True)

            # ── Calories ──
            total_calories = 0
            for item in bill["items"]:
                food = item["food"]
                if food in FOOD_INFO:
                    total_calories += FOOD_INFO[food]["calories"]

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="
                background:linear-gradient(135deg,#434291,#8083D6);
                border-radius:20px; padding:24px 32px; text-align:center;
                box-shadow:0 8px 24px rgba(67,66,145,0.25); margin-bottom:20px;
            ">
                <div style="font-size:13px; color:rgba(255,255,255,0.75); font-weight:600; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:6px;">🔥 Tổng Calories</div>
                <div style="font-size:42px; font-weight:800; color:#FFFFFF;">{total_calories}</div>
                <div style="font-size:16px; color:rgba(255,255,255,0.8); font-weight:500;">kcal</div>
            </div>
            """, unsafe_allow_html=True)

            # ════════════════════════════════════════════════════════
            # TÍNH NĂNG 1 — BEHAVIORAL NUDGE
            # ════════════════════════════════════════════════════════
            st.markdown("""<hr>""", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:15px; font-weight:700; color:#434291; margin-bottom:12px;">🥗 Nhận xét dinh dưỡng</div>
            """, unsafe_allow_html=True)

            food_names = [item["food"] for item in bill["items"]]
            has_vegetable = "Rau xào" in food_names or "Canh rau" in food_names

            if has_vegetable:
                st.markdown("""
                <div style="
                    background:linear-gradient(135deg,#f0fce8,#d8f5b0);
                    border-left:4px solid #9CD254;
                    border-radius:16px; padding:20px 24px;
                    box-shadow:0 2px 10px rgba(156,210,84,0.15);
                ">
                    <div style="font-size:22px; font-weight:800; color:#2C2B5E; margin-bottom:6px;">🎉 Tuyệt vời!</div>
                    <div style="font-size:15px; font-weight:600; color:#3a7d1e; margin-bottom:4px;">Bữa ăn rất cân bằng dinh dưỡng.</div>
                    <div style="font-size:14px; color:#4a9e26; font-weight:500;">Chúc bạn một ngày năng suất và khỏe mạnh! 💪</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="
                    background:linear-gradient(135deg,#fffbea,#fef3cc);
                    border-left:4px solid #f5c842;
                    border-radius:16px; padding:20px 24px;
                    box-shadow:0 2px 10px rgba(245,200,66,0.15);
                ">
                    <div style="font-size:22px; font-weight:800; color:#2C2B5E; margin-bottom:6px;">🥦 Gợi ý dinh dưỡng</div>
                    <div style="font-size:14px; font-weight:500; color:#7a5c00;">
                        Thêm một phần <strong>rau xào</strong> hoặc <strong>canh rau</strong> để bổ sung
                        vitamin và chất xơ cho buổi chiều năng suất nhé!
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ════════════════════════════════════════════════════════
            # TÍNH NĂNG 4 — GAMIFICATION: ĐIỂM SỨC KHỎE
            # ════════════════════════════════════════════════════════
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:15px; font-weight:700; color:#434291; margin-bottom:12px;">🏆 Điểm sức khỏe bữa ăn</div>
            """, unsafe_allow_html=True)

            health_score = 0
            if "Rau xào" in food_names:
                health_score += 20
            if "Canh rau" in food_names:
                health_score += 20
            if "Cá hú kho" in food_names:
                health_score += 15
            if "Đậu hũ sốt cà" in food_names:
                health_score += 10
            health_score = min(health_score, 100)

            if health_score >= 90:
                health_label = "🌟 Xuất sắc"
                health_color = "#9CD254"
                health_bg = "linear-gradient(135deg,#f0fce8,#d8f5b0)"
                health_border = "#9CD254"
            elif health_score >= 70:
                health_label = "✅ Tốt"
                health_color = "#434291"
                health_bg = "linear-gradient(135deg,#ededfb,#dcdeff)"
                health_border = "#8083D6"
            elif health_score >= 50:
                health_label = "🙂 Khá ổn"
                health_color = "#8083D6"
                health_bg = "linear-gradient(135deg,#f4f5fb,#e8eaf6)"
                health_border = "#8083D6"
            else:
                health_label = "⚠️ Cần bổ sung rau xanh"
                health_color = "#c97a00"
                health_bg = "linear-gradient(135deg,#fffbea,#fef3cc)"
                health_border = "#f5c842"

            st.markdown(f"""
            <div style="
                background:{health_bg};
                border:1.5px solid {health_border};
                border-radius:20px; padding:24px 28px;
                box-shadow:0 2px 12px rgba(128,131,214,0.10);
                margin-bottom:8px;
            ">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; flex-wrap:wrap; gap:8px;">
                    <div style="font-size:15px; font-weight:700; color:#2C2B5E;">Điểm sức khỏe</div>
                    <div style="
                        font-size:28px; font-weight:800; color:{health_color};
                    ">{health_score}<span style="font-size:14px; font-weight:600; color:#8083D6;">/100</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(health_score / 100)

            st.markdown(f"""
            <div style="
                display:inline-block; margin-top:10px;
                background:{health_color}22; border:1.5px solid {health_border};
                border-radius:50px; padding:6px 20px;
                font-size:14px; font-weight:700; color:{health_color};
            ">{health_label}</div>
            """, unsafe_allow_html=True)

            # ════════════════════════════════════════════════════════
            # TÍNH NĂNG 2 — AI FEEDBACK
            # ════════════════════════════════════════════════════════
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""<hr>""", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:15px; font-weight:700; color:#434291; margin-bottom:12px;">🤖 Đánh giá kết quả AI</div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="
                background:#F4F5FB; border:1.5px solid #8083D6;
                border-radius:16px; padding:20px 24px; margin-bottom:12px;
                box-shadow:0 2px 10px rgba(128,131,214,0.10);
            ">
                <div style="font-size:14px; font-weight:600; color:#2C2B5E; margin-bottom:14px;">
                    AI nhận diện kết quả trên có chính xác không?
                </div>
            </div>
            """, unsafe_allow_html=True)

            fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 3])

            with fb_col1:
                if st.button("👍 AI nhận diện đúng", key="feedback_good"):
                    st.session_state.feedback_given = "good"

            with fb_col2:
                if st.button("👎 AI nhận diện sai", key="feedback_bad"):
                    st.session_state.feedback_given = "bad"
                    os.makedirs("feedback_wrong", exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join("feedback_wrong", f"wrong_{ts}.jpg")
                    cv2.imwrite(save_path, st.session_state.image)

            if st.session_state.feedback_given == "good":
                st.markdown("""
                <div style="
                    background:linear-gradient(135deg,#f0fce8,#d8f5b0);
                    border-left:4px solid #9CD254;
                    border-radius:12px; padding:14px 20px; margin-top:10px;
                    color:#3a7d1e; font-weight:600; font-size:14px;
                ">✅ Cảm ơn bạn đã đánh giá! Phản hồi của bạn giúp chúng tôi cải thiện AI.</div>
                """, unsafe_allow_html=True)
            elif st.session_state.feedback_given == "bad":
                st.markdown("""
                <div style="
                    background:linear-gradient(135deg,#ededfb,#dcdeff);
                    border-left:4px solid #8083D6;
                    border-radius:12px; padding:14px 20px; margin-top:10px;
                    color:#434291; font-weight:600; font-size:14px;
                ">📸 Ảnh đã được lưu để huấn luyện lại AI. Cảm ơn bạn đã đóng góp!</div>
                """, unsafe_allow_html=True)

            # ════════════════════════════════════════════════════════
            # TÍNH NĂNG 3 — MÃ GIẢM GIÁ
            # ════════════════════════════════════════════════════════
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""<hr>""", unsafe_allow_html=True)
            st.markdown("""
            <div style="
                background:#F4F5FB; border:1.5px solid #8083D6;
                border-radius:20px; padding:24px 28px;
                box-shadow:0 2px 12px rgba(128,131,214,0.10);
            ">
                <div style="font-size:15px; font-weight:700; color:#434291; margin-bottom:4px;">🎁 Mã giảm giá</div>
                <div style="font-size:13px; color:#8083D6; font-weight:500; margin-bottom:16px;">Nhập mã ưu đãi nếu có (VD: MAMMAM10, UEH20, NHOM7)</div>
            </div>
            """, unsafe_allow_html=True)

            promo_col1, promo_col2 = st.columns([3, 1])
            with promo_col1:
                promo_code = st.text_input(
                    "Mã giảm giá",
                    placeholder="Nhập mã tại đây...",
                    label_visibility="collapsed"
                )
            with promo_col2:
                apply_promo = st.button("Áp dụng", key="apply_promo")

            if apply_promo:
                code_upper = promo_code.strip().upper()
                if code_upper in DISCOUNT_CODES:
                    st.session_state.promo_applied = code_upper
                    st.session_state.promo_discount_rate = DISCOUNT_CODES[code_upper]
                else:
                    st.session_state.promo_applied = "invalid"
                    st.session_state.promo_discount_rate = 0.0

            base_total = bill["total"] + extra_egg_cost
            discount_rate = st.session_state.promo_discount_rate
            discount_amount = int(base_total * discount_rate)
            final_total = base_total - discount_amount

            if st.session_state.promo_applied and st.session_state.promo_applied != "invalid":
                st.markdown(f"""
                <div style="
                    background:linear-gradient(135deg,#f0fce8,#d8f5b0);
                    border-left:4px solid #9CD254;
                    border-radius:12px; padding:14px 20px; margin-top:10px;
                    color:#2C2B5E; font-size:14px;
                ">
                    ✅ <strong>Áp dụng thành công mã {st.session_state.promo_applied}!</strong><br>
                    <span style="color:#3a7d1e;">Giảm {int(discount_rate*100)}% — tiết kiệm {discount_amount:,}đ</span>
                </div>
                """, unsafe_allow_html=True)
            elif st.session_state.promo_applied == "invalid":
                st.markdown("""
                <div style="
                    background:linear-gradient(135deg,#fff0f0,#ffe0e0);
                    border-left:4px solid #e05555;
                    border-radius:12px; padding:14px 20px; margin-top:10px;
                    color:#c0392b; font-size:14px; font-weight:600;
                ">❌ Mã giảm giá không hợp lệ. Vui lòng kiểm tra lại.</div>
                """, unsafe_allow_html=True)

            # ── Thanh toán ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""<hr>""", unsafe_allow_html=True)

            col1, col2 = st.columns([3, 2])

            with col1:
                if discount_amount > 0:
                    st.markdown(f"""
                    <div style="
                        background:#F4F5FB; border:1.5px solid #8083D6;
                        border-radius:20px; padding:32px 28px;
                        box-shadow:0 4px 16px rgba(128,131,214,0.12); height:100%;
                    ">
                        <div style="font-size:13px; color:#8083D6; font-weight:700; text-transform:uppercase; letter-spacing:0.6px; margin-bottom:8px;">💰 Chi tiết hóa đơn</div>
                        <div style="font-size:16px; color:#2C2B5E; margin-bottom:4px;">Tiền món ăn: <strong>{base_total:,}đ</strong></div>
                        <div style="font-size:16px; color:#9CD254; margin-bottom:12px; font-weight:600;">Giảm giá: <strong>-{discount_amount:,}đ</strong></div>
                        <div style="height:1.5px; background:linear-gradient(90deg,transparent,#8083D6,transparent); margin-bottom:14px;"></div>
                        <div style="font-size:13px; color:#8083D6; font-weight:700; text-transform:uppercase; letter-spacing:0.6px; margin-bottom:6px;">Tổng thanh toán</div>
                        <div style="font-size:44px; font-weight:800; color:#434291; line-height:1;">{final_total:,}<span style="font-size:22px;">đ</span></div>
                        <div style="
                            margin-top:16px; display:inline-block;
                            background:#9CD254; color:#2C2B5E;
                            padding:6px 18px; border-radius:50px;
                            font-size:12px; font-weight:700;
                        ">✅ Sẵn sàng thanh toán</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="
                        background:#F4F5FB; border:1.5px solid #8083D6;
                        border-radius:20px; padding:32px 28px;
                        box-shadow:0 4px 16px rgba(128,131,214,0.12); height:100%;
                    ">
                        <div style="font-size:13px; color:#8083D6; font-weight:700; text-transform:uppercase; letter-spacing:0.6px; margin-bottom:8px;">💰 Tổng hóa đơn</div>
                        <div style="font-size:48px; font-weight:800; color:#434291; line-height:1;">{final_total:,}<span style="font-size:24px;">đ</span></div>
                        <div style="
                            margin-top:16px; display:inline-block;
                            background:#9CD254; color:#2C2B5E;
                            padding:6px 18px; border-radius:50px;
                            font-size:12px; font-weight:700;
                        ">✅ Sẵn sàng thanh toán</div>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div style="
                    background:#F4F5FB; border:1.5px solid #8083D6;
                    border-radius:20px; padding:20px;
                    box-shadow:0 4px 16px rgba(128,131,214,0.12);
                    text-align:center;
                ">
                    <div style="font-size:13px; color:#8083D6; font-weight:700; text-transform:uppercase; letter-spacing:0.6px; margin-bottom:12px;">📱 Quét mã QR</div>
                """, unsafe_allow_html=True)
                qr_url = generate_vietqr(final_total)
                st.image(
                    qr_url,
                    caption="Quét mã để thanh toán",
                    use_container_width=True
                )
                st.markdown("""
                    <div style="font-size:11px; color:#8083D6; font-weight:500; margin-top:8px;">Hỗ trợ tất cả ứng dụng ngân hàng</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Lịch sử hóa đơn ──
            st.markdown("""<hr>""", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:15px; font-weight:700; color:#434291; margin-bottom:12px;">📜 Lịch sử hóa đơn</div>
            """, unsafe_allow_html=True)

            if os.path.exists("history.csv"):
                history_df = pd.read_csv("history.csv")
                st.dataframe(
                    history_df,
                    use_container_width=True
                )

# ─── PAGE: DASHBOARD ──────────────────────────────────────────────────────────
elif page == "📊 Dashboard":

    st.markdown("""
    <div style="
        padding: 28px 36px;
        border-radius: 20px;
        background: linear-gradient(135deg, #434291, #8083D6);
        margin-bottom: 28px;
        box-shadow: 0 8px 24px rgba(67,66,145,0.25);
    ">
        <h2 style="color:#FFFFFF; margin:0 0 6px 0; font-size:26px; font-weight:800;">📊 Dashboard</h2>
        <p style="color:rgba(255,255,255,0.75); margin:0; font-size:14px; font-weight:500;">
            Thống kê doanh thu và lịch sử giao dịch hệ thống
        </p>
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists("history.csv"):

        df = pd.read_csv("history.csv")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div style="
                background:#F4F5FB; border:1.5px solid #8083D6;
                border-radius:20px; padding:24px 20px; text-align:center;
                box-shadow:0 2px 12px rgba(128,131,214,0.12);
            ">
                <div style="font-size:32px; margin-bottom:8px;">🧾</div>
                <div style="font-size:36px; font-weight:800; color:#434291;">{len(df)}</div>
                <div style="font-size:12px; font-weight:700; color:#8083D6; text-transform:uppercase; letter-spacing:0.5px; margin-top:4px;">Số giao dịch</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="
                background:#F4F5FB; border:1.5px solid #8083D6;
                border-radius:20px; padding:24px 20px; text-align:center;
                box-shadow:0 2px 12px rgba(128,131,214,0.12);
            ">
                <div style="font-size:32px; margin-bottom:8px;">💰</div>
                <div style="font-size:30px; font-weight:800; color:#434291;">{df['total'].sum():,}đ</div>
                <div style="font-size:12px; font-weight:700; color:#8083D6; text-transform:uppercase; letter-spacing:0.5px; margin-top:4px;">Tổng doanh thu</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            avg = int(df['total'].mean()) if len(df) > 0 else 0
            st.markdown(f"""
            <div style="
                background:#F4F5FB; border:1.5px solid #8083D6;
                border-radius:20px; padding:24px 20px; text-align:center;
                box-shadow:0 2px 12px rgba(128,131,214,0.12);
            ">
                <div style="font-size:32px; margin-bottom:8px;">📈</div>
                <div style="font-size:30px; font-weight:800; color:#434291;">{avg:,}đ</div>
                <div style="font-size:12px; font-weight:700; color:#8083D6; text-transform:uppercase; letter-spacing:0.5px; margin-top:4px;">Trung bình / giao dịch</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:15px; font-weight:700; color:#434291; margin-bottom:10px;">📈 Biểu đồ doanh thu</div>
        """, unsafe_allow_html=True)

        st.line_chart(df["total"])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:15px; font-weight:700; color:#434291; margin-bottom:10px;">📋 Lịch sử chi tiết</div>
        """, unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

    else:
        st.markdown("""
        <div style="
            background:#F4F5FB; border:1.5px dashed #8083D6;
            border-radius:20px; padding:60px 40px; text-align:center;
            box-shadow:0 2px 12px rgba(128,131,214,0.08);
        ">
            <div style="font-size:48px; margin-bottom:16px;">📭</div>
            <div style="font-size:18px; font-weight:700; color:#434291; margin-bottom:8px;">Chưa có dữ liệu</div>
            <div style="font-size:14px; color:#8083D6; font-weight:500;">Hãy thực hiện nhận diện món ăn để bắt đầu ghi nhận lịch sử giao dịch.</div>
        </div>
        """, unsafe_allow_html=True)