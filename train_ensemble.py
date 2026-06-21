# ============================================================
# FOOD CNN — ENSEMBLE (EfficientNetB0 + B3 + ResNet50)
# Fix: dùng tf.data thay ImageDataGenerator (TF 2.20 / Keras 3)
# ============================================================

# ── 0. SETUP ────────────────────────────────────────────────

import os, shutil, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tf_keras
from tf_keras.applications import EfficientNetB0, EfficientNetB3, ResNet50
from tf_keras.applications.efficientnet import preprocess_input as eff_preprocess
from tf_keras.applications.resnet50 import preprocess_input as res_preprocess
from tf_keras.models import Model
from tf_keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tf_keras.optimizers import Adam
from tf_keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tf_keras.preprocessing.image import ImageDataGenerator
# QUAN TRỌNG: KHÔNG dùng mixed_float16 với Keras 3 + EfficientNet
tf.keras.mixed_precision.set_global_policy('float32')

print("TensorFlow:", tf.__version__)
print("Keras:", tf.keras.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))


# ── 1. CONFIG ────────────────────────────────────────────────
DATASET_PATH = "/content/drive/MyDrive/data/AI CUỐI KỲ"
OUTPUT_DIR   = "/content/dataset"
SAVE_DIR     = "/content/drive/MyDrive/ensemble_models"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE      = 32
EPOCHS_FREEZE   = 10
EPOCHS_UNFREEZE = 40
AUTOTUNE        = tf.data.AUTOTUNE

MODEL_CONFIGS = {
    "efficientnet_b0": {"size": 224, "preprocess": eff_preprocess},
    "efficientnet_b3": {"size": 300, "preprocess": eff_preprocess},
    "resnet50":        {"size": 224, "preprocess": res_preprocess},
}


# ── 2. KIỂM TRA DỮ LIỆU ─────────────────────────────────────
print("\n📁 Classes và số lượng ảnh:")
for folder in sorted(os.listdir(DATASET_PATH)):
    folder_path = os.path.join(DATASET_PATH, folder)
    if os.path.isdir(folder_path):
        print(f"  {folder:30s} → {len(os.listdir(folder_path))} ảnh")


# ── 3. CHIA TRAIN / VAL / TEST ───────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

for class_name in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_name)
    if not os.path.isdir(class_path):
        continue
    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]
    if len(images) < 2:
        continue
    train_imgs, temp = train_test_split(images, test_size=0.2, random_state=42)
    val_imgs, test_imgs = train_test_split(temp, test_size=0.5, random_state=42)
    for split, data in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        dst = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(dst, exist_ok=True)
        for img in data:
            shutil.copy(os.path.join(class_path, img), os.path.join(dst, img))

print("✅ Chia dữ liệu xong!")

# Lấy class names từ folder
CLASS_NAMES = sorted([
    d for d in os.listdir(os.path.join(OUTPUT_DIR, "train"))
    if os.path.isdir(os.path.join(OUTPUT_DIR, "train", d))
])
NUM_CLASSES = len(CLASS_NAMES)
CLASS_INDICES = {name: i for i, name in enumerate(CLASS_NAMES)}

print(f"📊 Số classes: {NUM_CLASSES}")
print(f"Classes: {CLASS_NAMES}")

with open(os.path.join(SAVE_DIR, "class_indices.json"), "w", encoding="utf-8") as f:
    json.dump(CLASS_INDICES, f, ensure_ascii=False, indent=2)
print("✅ Đã lưu class_indices.json")


# ── 4. HÀM TẠO tf.data PIPELINE ─────────────────────────────
def make_tf_dataset(split_dir, img_size, preprocess_fn, training=False):
    """
    Dùng tf.data thay ImageDataGenerator
    Tương thích hoàn toàn với Keras 3 / TF 2.20
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        image_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        shuffle=training,
        seed=42,
        label_mode="categorical"
    )

    def augment(image, label):
        if training:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.3)
            image = tf.image.random_contrast(image, 0.7, 1.3)
            image = tf.image.random_saturation(image, 0.7, 1.3)
        # Preprocess theo từng model
        image = tf.cast(image, tf.float32)
        image = preprocess_fn(image)
        return image, label

    dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def make_generators(img_size, preprocess_fn):
    train_ds = make_tf_dataset(
        os.path.join(OUTPUT_DIR, "train"), img_size, preprocess_fn, training=True
    )
    val_ds = make_tf_dataset(
        os.path.join(OUTPUT_DIR, "val"), img_size, preprocess_fn, training=False
    )
    test_ds = make_tf_dataset(
        os.path.join(OUTPUT_DIR, "test"), img_size, preprocess_fn, training=False
    )
    return train_ds, val_ds, test_ds


# ── 5. HÀM BUILD MODEL ───────────────────────────────────────
def build_model(base_model, num_classes):
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=out)


# ── 6. HÀM TRAIN 2 PHASE ─────────────────────────────────────
def train_two_phase(model, model_name, train_ds, val_ds, base_model):
    save_path = os.path.join(SAVE_DIR, f"{model_name}.h5")

    # Phase 1 — Freeze base, train top layers
    print(f"\n🔒 [{model_name}] Phase 1: Freeze ({EPOCHS_FREEZE} epochs)")
    model.compile(Adam(1e-3), "categorical_crossentropy", ["accuracy"])
    h1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_FREEZE,
        callbacks=[
            ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=0),
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)
        ],
        verbose=1
    )
    print(f"   ✅ val_acc: {max(h1.history['val_accuracy'])*100:.2f}%")

    # Phase 2 — Unfreeze 50 layers cuối
    print(f"\n🔓 [{model_name}] Phase 2: Fine-tune ({EPOCHS_UNFREEZE} epochs)")
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model.compile(Adam(1e-5), "categorical_crossentropy", ["accuracy"])
    h2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_UNFREEZE,
        callbacks=[
            ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=0),
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=0)
        ],
        verbose=1
    )
    print(f"   ✅ val_acc: {max(h2.history['val_accuracy'])*100:.2f}%")
    print(f"   💾 Saved: {save_path}")
    return h1, h2


# ── 7. TRAIN CẢ 3 MODELS ─────────────────────────────────────
histories     = {}
trained_models = {}
test_datasets  = {}

# EfficientNetB0
print("\n" + "="*55)
print("🚀 TRAINING 1/3 — EfficientNetB0")
print("="*55)
train_ds, val_ds, test_ds = make_generators(224, eff_preprocess)
base_b0  = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model_b0 = build_model(base_b0, NUM_CLASSES)
h1, h2   = train_two_phase(model_b0, "efficientnet_b0", train_ds, val_ds, base_b0)
histories["efficientnet_b0"]      = (h1, h2)
trained_models["efficientnet_b0"] = model_b0
test_datasets["efficientnet_b0"]  = test_ds

# EfficientNetB3
print("\n" + "="*55)
print("🚀 TRAINING 2/3 — EfficientNetB3")
print("="*55)
train_ds, val_ds, test_ds = make_generators(300, eff_preprocess)
base_b3  = EfficientNetB3(weights="imagenet", include_top=False, input_shape=(300, 300, 3))
model_b3 = build_model(base_b3, NUM_CLASSES)
h1, h2   = train_two_phase(model_b3, "efficientnet_b3", train_ds, val_ds, base_b3)
histories["efficientnet_b3"]      = (h1, h2)
trained_models["efficientnet_b3"] = model_b3
test_datasets["efficientnet_b3"]  = test_ds

# ResNet50
print("\n" + "="*55)
print("🚀 TRAINING 3/3 — ResNet50")
print("="*55)
train_ds, val_ds, test_ds = make_generators(224, res_preprocess)
base_res  = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model_res = build_model(base_res, NUM_CLASSES)
h1, h2    = train_two_phase(model_res, "resnet50", train_ds, val_ds, base_res)
histories["resnet50"]      = (h1, h2)
trained_models["resnet50"] = model_res
test_datasets["resnet50"]  = test_ds


# ── 8. ĐÁNH GIÁ TỪNG MODEL ───────────────────────────────────
print("\n📊 Đánh giá từng model trên Test set:")
individual_accs = {}
for name, mdl in trained_models.items():
    loss, acc = mdl.evaluate(test_datasets[name], verbose=0)
    individual_accs[name] = acc
    print(f"   {name:22s} → {acc*100:.2f}%")


# ── 9. ENSEMBLE ACCURACY ─────────────────────────────────────
print("\n🔀 Tính Ensemble accuracy (Soft Voting)...")

def get_preds_and_labels(model, dataset):
    preds, labels = [], []
    for x, y in dataset:
        preds.append(model.predict(x, verbose=0))
        labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)

b0_preds,  true_labels = get_preds_and_labels(model_b0,  test_datasets["efficientnet_b0"])
b3_preds,  _           = get_preds_and_labels(model_b3,  test_datasets["efficientnet_b3"])
res_preds, _           = get_preds_and_labels(model_res, test_datasets["resnet50"])

ensemble_preds = np.mean([b0_preds, b3_preds, res_preds], axis=0)
true_label_idx = np.argmax(true_labels, axis=1)
ensemble_acc   = np.mean(np.argmax(ensemble_preds, axis=1) == true_label_idx)

print(f"\n🏆 Kết quả:")
print(f"   EfficientNetB0   : {individual_accs['efficientnet_b0']*100:.2f}%")
print(f"   EfficientNetB3   : {individual_accs['efficientnet_b3']*100:.2f}%")
print(f"   ResNet50         : {individual_accs['resnet50']*100:.2f}%")
print(f"   {'─'*32}")
print(f"   🔀 Ensemble (avg): {ensemble_acc*100:.2f}%")


# ── 10. BIỂU ĐỒ ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for i, name in enumerate(["efficientnet_b0", "efficientnet_b3", "resnet50"]):
    h1, h2 = histories[name]
    acc_all      = h1.history["accuracy"]     + h2.history["accuracy"]
    val_acc_all  = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss_all     = h1.history["loss"]         + h2.history["loss"]
    val_loss_all = h1.history["val_loss"]     + h2.history["val_loss"]
    split = len(h1.history["accuracy"])

    axes[0][i].plot(acc_all, label="Train")
    axes[0][i].plot(val_acc_all, label="Val")
    axes[0][i].axvline(x=split, color="gray", linestyle="--", alpha=0.6, label="Fine-tune")
    axes[0][i].set_title(f"{name}\nAccuracy")
    axes[0][i].legend(); axes[0][i].grid(True)

    axes[1][i].plot(loss_all, label="Train")
    axes[1][i].plot(val_loss_all, label="Val")
    axes[1][i].axvline(x=split, color="gray", linestyle="--", alpha=0.6)
    axes[1][i].set_title(f"{name}\nLoss")
    axes[1][i].legend(); axes[1][i].grid(True)

plt.suptitle(f"Ensemble — Accuracy: {ensemble_acc*100:.2f}%", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "ensemble_training_history.png"), dpi=150)
plt.show()
print(f"✅ Hoàn tất! Models lưu tại: {SAVE_DIR}")