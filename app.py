import streamlit as st
import cv2
import tempfile
import pandas as pd
import os
import numpy as np
from datetime import datetime
from crop import main
from predict import run_prediction_pipeline

FOOD_INFO = {

    "Cơm trắng": {
        "calories": 250
    },

    "Sườn nướng": {
        "calories": 350
    },

    "Canh rau": {
        "calories": 40
    },

    "Canh chua": {
        "calories": 80
    },

    "Rau xào": {
        "calories": 120
    },

    "Thịt kho": {
        "calories": 280
    },

    "Thịt kho trứng": {
        "calories": 350
    },

    "Trứng chiên": {
        "calories": 180
    },

    "Đậu hũ sốt cà": {
        "calories": 150
    },

    "Cá hú kho": {
        "calories": 300
    }
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
if "image" not in st.session_state:
    st.session_state.image = None

if "original_image" not in st.session_state:
    st.session_state.original_image = None

st.sidebar.markdown("""
# 🍽️ CANTEEN AI

### Smart Food Recognition
""")

page = st.sidebar.radio(
    "📋 Menu",
    [
        "🏠 Trang chủ",
        "🍽️ Nhận diện món ăn",
        "📊 Dashboard"
    ]
)

if page == "🏠 Trang chủ":

    st.markdown("""
    <div style="
    padding:40px;
    border-radius:20px;
    background:linear-gradient(135deg,#1e293b,#0f172a);
    text-align:center;
    ">

    <h1 style="color:white;">
    🍽️ CANTEEN AI
    </h1>

    <h3 style="color:#cbd5e1;">
    Hệ thống nhận diện món ăn và tính hóa đơn tự động
    </h3>

    <p style="color:#94a3b8;">
    Upload hoặc chụp ảnh khay cơm để AI nhận diện món ăn,
    tính tiền và tạo mã QR thanh toán.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric(
            "🍚 Số món hỗ trợ",
            "10"
        )

    with col2:
        st.metric(
            "🤖 AI Model",
            "Ensemble CNN"
        )

    with col3:
        st.metric(
            "💳 Thanh toán",
            "VietQR"
        )

    st.divider()

    st.info("""
    🚀 Chọn mục 'Nhận diện món ăn' ở menu bên trái để bắt đầu sử dụng hệ thống.
    """)

elif page == "🍽️ Nhận diện món ăn":

    st.markdown("""
    <div style="
    padding:20px;
    border-radius:15px;
    background:#1e293b;
    ">

    <h2>
    📸 Nhận diện món ăn
    </h2>

    <p>
    Upload hoặc chụp ảnh khay cơm để AI phân tích.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.subheader("📸 Chọn nguồn ảnh")
    tab1, tab2 = st.tabs(["📁 Upload", "📷 Camera"])

    uploaded_file = None
    camera_file = None

    with tab1:
        uploaded_file = st.file_uploader(
            "Upload ảnh khay cơm",
            type=["jpg", "jpeg", "png"]
        )

    with tab2:
        camera_file = st.camera_input(
            "Chụp ảnh khay cơm"
        )

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

        st.subheader("🖼️ Ảnh xem trước")

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

        st.image(
            cv2.cvtColor(
                st.session_state.image,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )

        if st.button("🚀 Bắt đầu nhận diện"):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            ) as tmp:

                cv2.imwrite(
                    tmp.name,
                    st.session_state.image
                )

                image_path = tmp.name

            st.info("Đang xử lý ảnh...")

            rotated_img, compartments = main(
                use_demo=False,
                image_path=image_path,
                run_calibration=False
            )
            bill = run_prediction_pipeline(compartments)
            save_bill_history(bill)

            st.success("Nhận diện hoàn tất!")

            # Ảnh sau xoay
            st.image(
                cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB),
                caption="Ảnh sau xoay",
                use_container_width=True
            )

            # 5 ngăn crop
            st.subheader("🖼️ 5 ngăn sau khi crop")

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

                    st.image(
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                        caption=slot_names.get(slot, slot),
                        use_container_width=True
                    )

            # Kết quả
            st.subheader("Kết quả nhận diện")

            for item in bill["items"]:

                st.markdown(f"""
            ### {item['label']}

            **Món ăn:** {item['food']}

            **Độ tin cậy:** {item['confidence']:.2%}

            **Giá:** {item['price']:,}đ
            """)

                if item["note"]:
                    st.warning(item["note"])

            # Calories
            FOOD_INFO = {
                "Cơm trắng": 250,
                "Sườn nướng": 350,
                "Canh rau": 40,
                "Canh chua": 80,
                "Rau xào": 120,
                "Thịt kho": 280,
                "Thịt kho trứng": 350,
                "Trứng chiên": 180,
                "Đậu hũ sốt cà": 150,
                "Cá hú kho": 300
            }

            total_calories = 0

            for item in bill["items"]:

                food = item["food"]

                if food in FOOD_INFO:
                    total_calories += FOOD_INFO[food]

            st.metric(
                "🔥 Tổng Calories",
                f"{total_calories} kcal"
            )

            # Thanh toán
            st.divider()

            col1, col2 = st.columns([2,1])

            with col1:

                st.metric(
                    "💰 Tổng hóa đơn",
                    f"{bill['total']:,}đ"
                )

            with col2:

                qr_url = generate_vietqr(
                    bill["total"]
                )

                st.image(
                    qr_url,
                    caption="Quét mã để thanh toán",
                    use_container_width=True
                )

            # Lịch sử
            st.divider()

            st.subheader("📜 Lịch sử hóa đơn")

            if os.path.exists("history.csv"):

                history_df = pd.read_csv("history.csv")

                st.dataframe(
                    history_df,
                    use_container_width=True
                )

elif page == "📊 Dashboard":

    st.title("📊 Dashboard")

    if os.path.exists("history.csv"):

        df = pd.read_csv("history.csv")

        col1,col2 = st.columns(2)

        with col1:

            st.metric(
                "Số giao dịch",
                len(df)
            )

        with col2:

            st.metric(
                "Doanh thu",
                f"{df['total'].sum():,}đ"
            )

        st.line_chart(df["total"])