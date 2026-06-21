"""
╔══════════════════════════════════════════════════════════════════╗
║   NHẬN DIỆN MÓN ĂN & TÍNH HÓA ĐƠN KHAY CƠM CĂN TIN            ║
║   Pipeline: Ensemble (B0 + B3 + ResNet50) → Soft Voting → Bill  ║
║   Môi trường: Google Colab                                        ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── 0. IMPORTS ────────────────────────────────────────────────────
import os
import json
import numpy as np
import tensorflow as tf
import tf_keras
from tf_keras.applications.efficientnet import preprocess_input as eff_preprocess
from tf_keras.applications.resnet50      import preprocess_input as res_preprocess
import gdown
import re
import unicodedata

def download_models():

    os.makedirs("models", exist_ok=True)

    files = {
        "resnet50.h5":
            "1PXhYwq-N2v0qgxF-BhybGpPbR-GczcLT",

        "efficientnet_b0.h5":
            "1vDeX51hxItCUEFGrFEG9GmHI91vRnDOD",

        "efficientnet_b3.h5":
            "1yZ1OwdkQnHeV8_rKDBLml2M_kWdGFKiP",

        "class_indices.json":
            "1fPMmqXo6sU3-WRGR0nsVzI2h7bvXrwVS"
    }

    for filename, file_id in files.items():

        save_path = os.path.join("models", filename)

        if not os.path.exists(save_path):

            print(f"⬇️ Đang tải {filename} ...")

            url = f"https://drive.google.com/uc?id={file_id}"

            gdown.download(
                url,
                save_path,
                quiet=False
            )

            print(f"✅ Hoàn tất: {filename}")

# ══════════════════════════════════════════════════════════════════
#  PHẦN CẤU HÌNH — Chỉnh đường dẫn theo thực tế của bạn
# ══════════════════════════════════════════════════════════════════

SAVE_DIR = os.path.join(
    os.path.dirname(__file__),
    "models"
) 

MODEL_PATHS = {
    "efficientnet_b0": os.path.join(SAVE_DIR, "efficientnet_b0.h5"),
    "efficientnet_b3": os.path.join(SAVE_DIR, "efficientnet_b3.h5"),
    "resnet50":        os.path.join(SAVE_DIR, "resnet50.h5"),
}

CLASS_INDICES_PATH = os.path.join(SAVE_DIR, "class_indices.json")

# Kích thước ảnh đầu vào cho từng model
INPUT_SIZES = {
    "efficientnet_b0": 224,
    "efficientnet_b3": 300,
    "resnet50":        224,
}

# Hàm preprocess tương ứng
PREPROCESS_FNS = {
    "efficientnet_b0": eff_preprocess,
    "efficientnet_b3": eff_preprocess,
    "resnet50":        res_preprocess,
}

# ── MENU GIÁ MÓN ĂN ──────────────────────────────────────────────
MENU = {
    "Cơm trắng": {
        "price": 10_000
    },
    "Đậu hũ sốt cà": {
        "price": 25_000
    },
    "Cá hú kho": {
        "price": 30_000
    },
    "Thịt kho trứng": {
        "price": 30_000,
        "note": "Giá áp dụng cho 1 trứng. Mỗi trứng thêm +6.000đ."
    },
    "Thịt kho": {
        "price": 25_000
    },
    "Canh chua": {
        "price": 25_000
    },
    "Canh rau": {
        "price": 10_000
    },
    "Sườn nướng": {
        "price": 30_000
    },
    "Rau xào": {
        "price": 10_000
    },
    "Trứng chiên": {
        "price": 25_000
    },
}

# Nhãn hiển thị thân thiện cho từng ngăn
SLOT_LABELS = {
    "soup" : "Canh",
    "rice" : "Cơm",
    "comp1": "Món phụ 1",
    "comp2": "Món phụ 2",
    "comp3": "Món phụ 3",
}


# ══════════════════════════════════════════════════════════════════
#  BƯỚC 1 · TẢI MODEL VÀ CLASS INDICES
# ══════════════════════════════════════════════════════════════════

def load_models(model_paths: dict) -> dict:
    """
    Tải 3 model đã train từ file .h5.

    Args:
        model_paths (dict): {'model_name': '/path/to/model.h5', ...}

    Returns:
        dict: {'model_name': tf_keras.Model, ...}
    """
    models = {}
    for name, path in model_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[load_models] Không tìm thấy model: {path}\n"
                f"  → Kiểm tra lại SAVE_DIR và tên file .h5"
            )
        print(f"  📦 Đang tải {name:22s} ← {path}")
        models[name] = tf_keras.models.load_model(path, compile=False)
        print(f"      ✓ Tải xong  |  input shape: {models[name].input_shape}")
    return models


def load_class_indices(path: str) -> tuple:
    """
    Đọc file class_indices.json.

    Returns:
        (class_to_idx, idx_to_class)
        Ví dụ:
            class_to_idx = {'Cơm trắng': 0, 'Đậu hũ sốt cà': 1, ...}
            idx_to_class = {0: 'Cơm trắng', 1: 'Đậu hũ sốt cà', ...}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[load_class_indices] Không tìm thấy: {path}\n"
            f"  → Kiểm tra lại CLASS_INDICES_PATH"
        )
    with open(path, "r", encoding="utf-8") as f:
        class_to_idx: dict = json.load(f)

    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    print(f"  📋 class_indices.json: {len(class_to_idx)} classes")
    return class_to_idx, idx_to_class


# ══════════════════════════════════════════════════════════════════
#  BƯỚC 2 · TIỀN XỬ LÝ ẢNH CHO TỪNG MODEL
# ══════════════════════════════════════════════════════════════════

def preprocess_for_model(image_bgr: np.ndarray,
                         target_size: int,
                         preprocess_fn) -> np.ndarray:
    """
    Chuẩn bị 1 ảnh crop (BGR, np.ndarray) cho inference:
        1. Resize về target_size × target_size
        2. BGR → RGB
        3. Áp dụng preprocess_fn tương ứng của từng backbone
        4. Thêm batch dimension → (1, H, W, 3)

    Args:
        image_bgr    (np.ndarray): ảnh crop BGR từ crop_compartments()
        target_size  (int)       : kích thước đầu vào model (224 hoặc 300)
        preprocess_fn            : hàm preprocess của Keras (eff/res)

    Returns:
        np.ndarray shape (1, target_size, target_size, 3)
    """
    # Resize
    img = tf.image.resize(image_bgr, [target_size, target_size])
    img = tf.cast(img, tf.float32)

    # BGR → RGB (OpenCV dùng BGR, Keras dùng RGB)
    img = img[..., ::-1]

    # Áp dụng preprocess của backbone
    img = preprocess_fn(img)

    # Thêm batch dim
    img = tf.expand_dims(img, axis=0)

    return img.numpy()


# ══════════════════════════════════════════════════════════════════
#  BƯỚC 3 · DỰ ĐOÁN 1 ẢNH BẰNG ENSEMBLE
# ══════════════════════════════════════════════════════════════════
def predict_single(image_bgr: np.ndarray,
                   models: dict,
                   idx_to_class: dict,
                   top_k: int = 3) -> dict:

    all_probs = []

    for name, model in models.items():
        size = INPUT_SIZES[name]
        preprocess = PREPROCESS_FNS[name]
        x = preprocess_for_model(image_bgr, size, preprocess)
        probs = model.predict(x, verbose=0)[0]
        all_probs.append(probs)

    final_probs = np.mean(all_probs, axis=0)

    top_indices = np.argsort(final_probs)[::-1][:top_k]

    top_results = [
        (idx_to_class[i], float(final_probs[i]))
        for i in top_indices
    ]

    CONFIDENCE_THRESHOLD = 0.70

    best_food, best_conf = top_results[0]

    if best_conf < CONFIDENCE_THRESHOLD:
        best_food = "Món ăn không có trong thực đơn"

    return {
        "food": best_food,
        "confidence": best_conf,
        "top3": top_results,
    }

# ══════════════════════════════════════════════════════════════════
#  BƯỚC 4 · DỰ ĐOÁN TẤT CẢ 5 NGĂN
# ══════════════════════════════════════════════════════════════════

def predict_all_compartments(compartments: dict,
                              models:       dict,
                              idx_to_class: dict) -> dict:
    """
    Chạy predict_single() cho cả 5 ngăn trong compartments.

    Args:
        compartments (dict): {'soup': np.ndarray, 'rice': ..., ...}
        models       (dict): {'efficientnet_b0': model, ...}
        idx_to_class (dict): {0: 'Cơm trắng', ...}

    Returns:
        dict: {
            'soup' : {'food': ..., 'confidence': ..., 'top3': [...]},
            'rice' : {...},
            'comp1': {...},
            'comp2': {...},
            'comp3': {...},
        }
    """
    results = {}
    for slot, image_bgr in compartments.items():
        label = SLOT_LABELS.get(slot, slot)
        print(f"  🔍 Dự đoán [{label:10s}] ... ", end="", flush=True)
        pred = predict_single(image_bgr, models, idx_to_class)
        results[slot] = pred
        print(f"{pred['food']}  ({pred['confidence']*100:.1f}%)")
    return results


# ══════════════════════════════════════════════════════════════════
#  BƯỚC 5 · TRA GIÁ VÀ TÍNH HÓA ĐƠN
# ══════════════════════════════════════════════════════════════════

def lookup_price(food_name: str, menu: dict):

    if food_name == "Món ăn không có trong thực đơn":
        return 0, "Không tính tiền"

    if food_name in menu:
        entry = menu[food_name]
        return entry["price"], entry.get("note", None)

    return 0, "Không tìm thấy trong MENU"

def normalize_name(name):
    name = re.sub(r'^\d+\.\s*', '', name).strip()
    return unicodedata.normalize('NFC', name)

def compute_bill(predictions: dict, menu: dict) -> dict:

    items       = []
    total       = 0
    not_in_menu = []

    for slot, pred in predictions.items():

        # BỎ SỐ THỨ TỰ PHÍA TRƯỚC
        food = normalize_name(pred["food"])

        price, note = lookup_price(food, menu)

        item = {
            "slot"      : slot,
            "label"     : SLOT_LABELS.get(slot, slot),
            "food"      : food,
            "confidence": pred["confidence"],
            "top3"      : pred["top3"],
            "price"     : price,
            "note"      : note,
        }

        items.append(item)

        if price >= 0:
            total += price
        else:
            not_in_menu.append(food)

    return {
        "items"      : items,
        "total"      : total,
        "not_in_menu": not_in_menu,
    }

# ══════════════════════════════════════════════════════════════════
#  BƯỚC 6 · IN KẾT QUẢ
# ══════════════════════════════════════════════════════════════════

def print_results(bill: dict) -> None:
    """
    In kết quả nhận diện và hóa đơn ra console theo định dạng đẹp.

    Args:
        bill (dict): kết quả từ compute_bill()
    """
    LINE = "─" * 62
    DLINE = "═" * 62

    print(f"\n{DLINE}")
    print(f"  🍽   KẾT QUẢ NHẬN DIỆN KHAY CƠM CĂN TIN")
    print(DLINE)

    for item in bill["items"]:
        slot_label  = item["label"]
        food        = item["food"]
        if food == "Món ăn không có trong thực đơn":
          print(f"  ⚠ Món ăn chưa được hỗ trợ bởi hệ thống")
        conf        = item["confidence"] * 100
        price       = item["price"]
        note        = item["note"]
        top3        = item["top3"]

        # ── Dòng chính ───────────────────────────────────────────
        price_str = f"{price:>10,}đ".replace(",", ".") if price >= 0 else "   Không có"
        print(f"\n  {'[' + slot_label + ']':14s}  {food}")
        print(f"  {'':14s}  Độ tin cậy  : {conf:.1f}%")
        print(f"  {'':14s}  Giá         : {price_str}")

        if note:
            print(f"  {'':14s}  ⚠  Ghi chú  : {note}")

        # ── Top-3 ứng cử viên ────────────────────────────────────
        print(f"  {'':14s}  Top-3 dự đoán:")
        for rank, (fname, fconf) in enumerate(top3, 1):
            bar    = "█" * int(fconf * 20)
            marker = "  ✓" if rank == 1 else "   "
            print(f"  {'':14s}    {rank}. {fname:<22s} {bar:<20s} {fconf*100:5.1f}%{marker}")

        print(f"  {LINE}")

    # ── Tổng hóa đơn ─────────────────────────────────────────────
    total_str = f"{bill['total']:,}đ".replace(",", ".")
    print(f"\n  {'TỔNG HÓA ĐƠN':>38s} :  {total_str}")

    # ── Cảnh báo món không có trong menu ─────────────────────────
    if bill["not_in_menu"]:
        print(f"\n  ⚠  Món chưa có trong MENU (chưa tính giá):")
        for food in bill["not_in_menu"]:
            print(f"      • {food}")

    print(f"\n{DLINE}\n")


# ══════════════════════════════════════════════════════════════════
#  HÀM TỔNG · GỌI TOÀN BỘ PIPELINE
# ══════════════════════════════════════════════════════════════════

# Tải model từ Google Drive nếu chưa có
download_models()

# Load model 1 lần duy nhất khi mở app
MODELS = load_models(MODEL_PATHS)
_, IDX_TO_CLASS = load_class_indices(CLASS_INDICES_PATH)

def run_prediction_pipeline(compartments: dict) -> dict:
    """
    Chạy toàn bộ pipeline nhận diện + tính hóa đơn.
    """

    print("=" * 62)
    print("  PIPELINE NHẬN DIỆN MÓN ĂN — ENSEMBLE SOFT VOTING")
    print("=" * 62)

    # Dùng model đã load sẵn
    models = MODELS
    idx_to_class = IDX_TO_CLASS

    # ── Bước 1 · Dự đoán 5 ngăn ───────────────────────────────
    print("\n── Bước 1 · Dự đoán từng ngăn (Ensemble) ───────────────")

    predictions = predict_all_compartments(
        compartments,
        models,
        idx_to_class
    )

    # ── Bước 2 · Tính hóa đơn ─────────────────────────────────
    print("\n── Bước 2 · Tra giá và tính hóa đơn ────────────────────")

    bill = compute_bill(
        predictions,
        MENU
    )

    # ── Bước 3 · In kết quả ──────────────────────────────────
    print_results(bill)

    return bill
