"""
╔══════════════════════════════════════════════════════════════════╗
║   NHẬN DIỆN & CROP NGĂN KHAY CƠM CĂN TIN                       ║
║   Pipeline: Ảnh đầu vào → Detect hướng → Xoay chuẩn → Crop     ║
║   Thư viện: OpenCV · NumPy · Matplotlib                          ║
║   Môi trường: Google Colab (chạy độc lập)                        ║
╚══════════════════════════════════════════════════════════════════╝

Cấu trúc khay 5 ngăn (sau khi xoay về hướng chuẩn):
┌──────────┬──────────┬────────────┐
│          │  COMP3   │    RICE    │
│  SOUP    │ (món phụ)│   (cơm)   │
│  (canh)  ├──────────┴────────────┤
│          │       COMP2           │
├──────────┤     (món phụ 2)       │
│  COMP1   │                       │
│ (món phụ)│                       │
└──────────┴───────────────────────┘
  ↑ Đây chỉ là ví dụ layout. Tọa độ thực tế chỉnh ở phần CONFIG.
"""

# ── Cài đặt (chỉ cần chạy một lần trên Colab) ─────────────────
# !pip install opencv-python-headless matplotlib numpy -q

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches


# ══════════════════════════════════════════════════════════════════
#  PHẦN CẤU HÌNH — Chỉnh các giá trị này theo ảnh thực tế của bạn
# ══════════════════════════════════════════════════════════════════
#
#  Tọa độ (x1, y1, x2, y2) đo trên ảnh ĐÃ XOAY về hướng chuẩn.
#  x tăng sang phải, y tăng xuống dưới.
#  Dùng hàm show_calibration_grid() bên dưới để tìm tọa độ đúng.
#
SOUP_BOX  = ( 353,  125, 886, 609)   # Ngăn canh    — thường góc trên-trái
RICE_BOX  = (1067,  126, 1479, 577)   # Ngăn cơm     — LUÔN ở góc TRÊN-PHẢI
COMP1_BOX = ( 384, 602, 677, 968)   # Món phụ 1    — giữa-trái
COMP2_BOX = (702, 626, 100, 970)   # Món phụ 2    — giữa-phải
COMP3_BOX = (1080, 635, 1428, 955)   # Món phụ 3    — cạnh trái ngăn cơm
# ══════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────
#  BƯỚC 0 · TẢI ẢNH
# ─────────────────────────────────────────────────────────────────

def load_image(image_path: str = None) -> np.ndarray:
    """
    Tải ảnh đầu vào:
    - Nếu chạy trong Google Colab và không truyền path → hiện hộp upload.
    - Nếu truyền image_path → đọc trực tiếp từ đường dẫn đó.

    Returns:
        np.ndarray: ảnh BGR (định dạng chuẩn của OpenCV)
    """
    # -- Trường hợp 1: đường dẫn file được cung cấp trực tiếp ------
    if image_path is not None:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
        print(f"[load_image] Đọc từ file: {image_path}  |  size: {img.shape}")
        return img

# ─────────────────────────────────────────────────────────────────
#  BƯỚC 1 · PHÁT HIỆN GÓC CHỨA NGĂN CƠM
# ─────────────────────────────────────────────────────────────────

def detect_rice_corner(image: np.ndarray,
                       debug: bool = True) -> str:
    """
    Xác định góc nào trong 4 góc của ảnh đang chứa ngăn cơm.

    Nguyên lý:
        Cơm trắng có đặc điểm màu sắc đặc trưng trong không gian HSV:
        • Hue (H)        : bất kỳ (0–180)
        • Saturation (S) : thấp  (0–65)   ← không bão hòa = gần trắng
        • Value (V)      : cao   (160–255) ← sáng

        Chia ảnh thành 4 phần tư (quadrant) và đếm số pixel thỏa
        điều kiện trên ở mỗi góc. Góc có nhiều nhất = ngăn cơm.

    Args:
        image (np.ndarray): ảnh BGR từ camera/file
        debug (bool)      : in thông tin chi tiết ra console

    Returns:
        str: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right'
    """
    h, w = image.shape[:2]

    # ── 1a. Chuyển BGR → HSV ────────────────────────────────────
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ── 1b. Tạo mask vùng màu trắng / gần trắng ─────────────────
    #       Ngưỡng có thể tinh chỉnh nếu cơm bị vàng hoặc ánh sáng thay đổi
    lower_white = np.array([0,   0,  160])   # (H_min, S_min, V_min)
    upper_white = np.array([180, 65, 255])   # (H_max, S_max, V_max)
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # ── 1c. Morphological — làm sạch nhiễu ──────────────────────
    #       CLOSE: lấp lỗ hổng nhỏ bên trong vùng trắng
    #       OPEN : xoá đảo pixel lẻ tẻ bên ngoài
    k      = np.ones((13, 13), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    # ── 1d. Chia thành 4 góc và đếm pixel trắng ─────────────────
    mh, mw = h // 2, w // 2   # tâm ảnh

    quadrants = {
        'top-left':     mask[:mh,  :mw ],
        'top-right':    mask[:mh,  mw: ],
        'bottom-left':  mask[mh:,  :mw ],
        'bottom-right': mask[mh:,  mw: ],
    }

    counts = {
        corner: int(np.count_nonzero(q))
        for corner, q in quadrants.items()
    }

    # ── 1e. Chọn góc có nhiều pixel trắng nhất ──────────────────
    rice_corner = max(counts, key=counts.get)
    total_max   = max(counts.values()) or 1   # tránh chia cho 0

    # ── 1f. Debug log ────────────────────────────────────────────
    if debug:
        print("\n┌─ detect_rice_corner() ──────────────────────────┐")
        for corner, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            bar  = '█' * int(cnt / total_max * 30)
            mark = '  ← NGĂN CƠM ✓' if corner == rice_corner else ''
            print(f"│  {corner:15s} {bar:<30s} {cnt:7,d} px{mark}")
        print(f"└─ Kết quả: ngăn cơm ở [{rice_corner}] ──────────┘\n")

    return rice_corner


# ─────────────────────────────────────────────────────────────────
#  BƯỚC 2 · XOAY VỀ HƯỚNG CHUẨN
# ─────────────────────────────────────────────────────────────────

def rotate_to_standard(image: np.ndarray,
                       rice_corner: str) -> np.ndarray:
    """
    Xoay ảnh sao cho ngăn cơm luôn nằm ở góc TRÊN-PHẢI (top-right).

    Bảng góc xoay (chiều kim đồng hồ / clockwise):
    ┌─────────────────┬─────────────┬──────────────────────────────────┐
    │ Vị trí cơm hiện│ Góc xoay    │ Ghi chú                          │
    ├─────────────────┼─────────────┼──────────────────────────────────┤
    │ top-right       │  0°         │ Đúng rồi, không làm gì           │
    │ top-left        │ 90° CW      │ top-left  → top-right            │
    │ bottom-left     │ 180°        │ Lật ngược                        │
    │ bottom-right    │ 90° CCW     │ bottom-right → top-right         │
    └─────────────────┴─────────────┴──────────────────────────────────┘

    Args:
        image       (np.ndarray): ảnh BGR gốc
        rice_corner (str)       : góc hiện tại chứa ngăn cơm

    Returns:
        np.ndarray: ảnh BGR đã xoay về hướng chuẩn
    """
    # Ánh xạ vị trí cơm → mã lệnh xoay của OpenCV
    # Mục tiêu: đưa ngăn cơm về góc TOP-RIGHT
    #
    # Xác minh từng trường hợp (xoay 90°CW: top-left→top-right, bottom-right→bottom-left;
    #                            xoay 90°CCW: bottom-right→top-right, top-left→bottom-left;
    #                            xoay 180°: bottom-left→top-right):
    rotation_codes = {
        'top-right':    None,                            # đã đúng vị trí, không xoay
        'top-left':     cv2.ROTATE_90_CLOCKWISE,         # 90° CW  → top-left  đến top-right
        'bottom-left':  cv2.ROTATE_180,                  # 180°    → bottom-left đến top-right
        'bottom-right': cv2.ROTATE_90_COUNTERCLOCKWISE,  # 90° CCW → bottom-right đến top-right
    }

    label = {
        'top-right':    '0°          (không đổi)',
        'top-left':     '90° CW      (xoay phải)',
        'bottom-left':  '180°        (lật ngược)',
        'bottom-right': '90° CCW     (xoay trái)',
    }

    code = rotation_codes.get(rice_corner)
    print(f"[rotate_to_standard] Áp dụng xoay: {label.get(rice_corner, '?')}")

    if code is None:
        # Không cần xoay — trả về bản sao để tránh thay đổi ngoài ý muốn
        return image.copy()

    rotated = cv2.rotate(image, code)
    print(f"[rotate_to_standard] Kích thước: {image.shape[1]}×{image.shape[0]}"
          f" → {rotated.shape[1]}×{rotated.shape[0]}")
    return rotated


# ─────────────────────────────────────────────────────────────────
#  BƯỚC 3 · CROP 5 NGĂN
# ─────────────────────────────────────────────────────────────────

def crop_compartments(rotated_image: np.ndarray) -> dict:
    """
    Crop 5 ngăn của khay theo tọa độ cố định đã khai báo ở phần CONFIG.

    Sử dụng các biến toàn cục:
        SOUP_BOX, RICE_BOX, COMP1_BOX, COMP2_BOX, COMP3_BOX
        Mỗi biến là tuple (x1, y1, x2, y2) — pixel trên ảnh đã xoay.

    Tính năng an toàn:
        - Tự động clamp tọa độ về trong biên ảnh.
        - Cảnh báo rõ khi box bị clamp hoặc không hợp lệ.

    Args:
        rotated_image (np.ndarray): ảnh BGR đã xoay về hướng chuẩn

    Returns:
        dict: {
            'soup'  : np.ndarray (BGR),
            'rice'  : np.ndarray (BGR),
            'comp1' : np.ndarray (BGR),
            'comp2' : np.ndarray (BGR),
            'comp3' : np.ndarray (BGR),
        }
    """
    img_h, img_w = rotated_image.shape[:2]

    def safe_crop(img: np.ndarray,
                  box: tuple,
                  name: str) -> np.ndarray:
        """
        Nội hàm: crop an toàn với kiểm tra biên và cảnh báo.
        """
        x1, y1, x2, y2 = box

        # Đảm bảo x1 < x2 và y1 < y2
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1

        # Clamp vào phạm vi ảnh
        cx1 = max(0, min(x1, img_w))
        cy1 = max(0, min(y1, img_h))
        cx2 = max(0, min(x2, img_w))
        cy2 = max(0, min(y2, img_h))

        # Cảnh báo nếu có clamp
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            print(f"  ⚠  {name}: clamp ({x1},{y1},{x2},{y2})"
                  f" → ({cx1},{cy1},{cx2},{cy2})")

        # Kiểm tra vùng crop có diện tích > 0
        if cx2 <= cx1 or cy2 <= cy1:
            print(f"  ✗  {name}: tọa độ không hợp lệ — trả về ảnh đen 80×80")
            return np.zeros((80, 80, 3), dtype=np.uint8)

        crop = img[cy1:cy2, cx1:cx2]
        print(f"  ✓  {name:10s}: ({cx1:4d},{cy1:4d}) → ({cx2:4d},{cy2:4d})"
              f"  [{crop.shape[1]}×{crop.shape[0]} px]")
        return crop

    # ── Thực hiện crop 5 ngăn ────────────────────────────────────
    print(f"\n[crop_compartments] Ảnh đã xoay: {img_w}×{img_h} px")
    crops = {
        'soup' : safe_crop(rotated_image, SOUP_BOX,  'SOUP_BOX '),
        'rice' : safe_crop(rotated_image, RICE_BOX,  'RICE_BOX '),
        'comp1': safe_crop(rotated_image, COMP1_BOX, 'COMP1_BOX'),
        'comp2': safe_crop(rotated_image, COMP2_BOX, 'COMP2_BOX'),
        'comp3': safe_crop(rotated_image, COMP3_BOX, 'COMP3_BOX'),
    }
    return crops


# ─────────────────────────────────────────────────────────────────
#  BƯỚC 4 · HIỂN THỊ KẾT QUẢ
# ─────────────────────────────────────────────────────────────────

def visualize_results(original:    np.ndarray,
                      rotated:     np.ndarray,
                      crops:       dict,
                      rice_corner: str,
                      save_path:   str = 'tray_result.png') -> None:
    """
    Vẽ figure tổng hợp toàn bộ pipeline:

    Hàng 1: [Ảnh gốc + nhãn góc cơm]  [Ảnh đã xoay + 5 bounding boxes]
    Hàng 2: [Canh]  [Cơm]  [Món phụ 1]  [Món phụ 2]  [Món phụ 3]

    Args:
        original    : ảnh BGR gốc chưa xử lý
        rotated     : ảnh BGR sau khi xoay
        crops       : dict 5 ảnh crop {'soup', 'rice', 'comp1', 'comp2', 'comp3'}
        rice_corner : chuỗi góc đã phát hiện (để ghi lên ảnh)
        save_path   : đường dẫn file PNG kết quả
    """

    # ── Helper: BGR → RGB cho Matplotlib ─────────────────────────
    def to_rgb(bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ── Màu đặc trưng cho từng ngăn ──────────────────────────────
    COLORS = {
        'soup' : '#00e5ff',   # cyan
        'rice' : '#ffeb3b',   # vàng
        'comp1': '#69f0ae',   # xanh lá
        'comp2': '#ff9800',   # cam
        'comp3': '#e040fb',   # tím
    }
    LABELS = {
        'soup' : 'Canh',
        'rice' : 'Cơm',
        'comp1': 'Món phụ 1',
        'comp2': 'Món phụ 2',
        'comp3': 'Món phụ 3',
    }

    # ── Tạo figure với GridSpec 2 hàng × 5 cột ───────────────────
    fig = plt.figure(figsize=(22, 10))
    fig.patch.set_facecolor('#12131a')  # nền tối

    gs = gridspec.GridSpec(
        2, 5, figure=fig,
        hspace=0.38, wspace=0.20,
        left=0.03, right=0.97,
        top=0.91, bottom=0.03,
    )

    fig.suptitle(
        "🍽  Pipeline Nhận Diện Ngăn Khay Cơm Căn Tin",
        fontsize=16, fontweight='bold',
        color='white', y=0.97,
    )

    # ──────────────────────────────────────────────────────────────
    #  HÀNG 1 — ẢNH GỐC (2 cột) & ẢNH ĐÃ XOAY + BOXES (3 cột)
    # ──────────────────────────────────────────────────────────────

    # ── Subplot A: Ảnh gốc ───────────────────────────────────────
    ax_orig = fig.add_subplot(gs[0, :2])
    ax_orig.imshow(to_rgb(original))
    ax_orig.set_title(
        f"① ẢNH GỐC  |  Phát hiện ngăn cơm ở: 【{rice_corner}】",
        color='white', fontsize=10, pad=7, loc='left',
    )
    ax_orig.axis('off')

    # Đánh dấu vị trí ngăn cơm trên ảnh gốc bằng nhãn nổi bật
    h_o, w_o = original.shape[:2]
    rice_indicator = {
        'top-left':     (w_o * 0.18, h_o * 0.18),
        'top-right':    (w_o * 0.82, h_o * 0.18),
        'bottom-left':  (w_o * 0.18, h_o * 0.82),
        'bottom-right': (w_o * 0.82, h_o * 0.82),
    }
    rx, ry = rice_indicator.get(rice_corner, (w_o / 2, h_o / 2))
    ax_orig.text(
        rx, ry, 'CƠM',
        fontsize=13, fontweight='bold', color='#ffeb3b',
        ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.45', facecolor='#12131a',
                  edgecolor='#ffeb3b', linewidth=2, alpha=0.85),
    )

    # ── Subplot B: Ảnh đã xoay + Bounding Boxes ──────────────────
    ax_rot = fig.add_subplot(gs[0, 2:])
    ax_rot.imshow(to_rgb(rotated))
    ax_rot.set_title(
        "② ẢNH ĐÃ XOAY  |  Cơm → góc trên-phải  |  5 bounding boxes",
        color='white', fontsize=10, pad=7, loc='left',
    )
    ax_rot.axis('off')

    # Vẽ bounding box và nhãn cho mỗi ngăn
    box_defs = [
        ('soup',  SOUP_BOX),
        ('rice',  RICE_BOX),
        ('comp1', COMP1_BOX),
        ('comp2', COMP2_BOX),
        ('comp3', COMP3_BOX),
    ]

    for key, (x1, y1, x2, y2) in box_defs:
        color = COLORS[key]
        label = LABELS[key]

        # Hình chữ nhật (fill mờ + viền)
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5,
            edgecolor=color,
            facecolor=color,
            alpha=0.18,
        )
        ax_rot.add_patch(rect)

        # Nhãn text góc trên-trái của box
        ax_rot.text(
            x1 + 6, y1 + 6, label,
            fontsize=8.5, fontweight='bold', color=color,
            va='top', ha='left',
            bbox=dict(facecolor='#12131a', alpha=0.65,
                      pad=2.5, boxstyle='round,pad=0.3'),
        )

    # ──────────────────────────────────────────────────────────────
    #  HÀNG 2 — 5 ẢNH CROP
    # ──────────────────────────────────────────────────────────────
    crop_order = ['soup', 'rice', 'comp1', 'comp2', 'comp3']
    boxes      = dict(zip(
        crop_order,
        [SOUP_BOX, RICE_BOX, COMP1_BOX, COMP2_BOX, COMP3_BOX]
    ))

    for col, key in enumerate(crop_order):
        ax = fig.add_subplot(gs[1, col])
        color  = COLORS[key]
        label  = LABELS[key]
        x1, y1, x2, y2 = boxes[key]
        size_s = f"{x2-x1}×{y2-y1} px"

        # Hiển thị ảnh crop
        ax.imshow(to_rgb(crops[key]))

        # Tiêu đề và kích thước
        ax.set_title(
            f"{label}\n{size_s}",
            color=color, fontsize=9.5, fontweight='bold', pad=5,
        )

        # Viền màu đặc trưng
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(3.0)

        # Số thứ tự góc phải-dưới
        ax.text(
            0.97, 0.03, f"#{col+1}",
            transform=ax.transAxes,
            fontsize=8, color=color, alpha=0.7,
            ha='right', va='bottom',
        )

    # ── Lưu và hiển thị ──────────────────────────────────────────
    plt.savefig(
        save_path, dpi=150,
        bbox_inches='tight',
        facecolor=fig.get_facecolor(),
    )
    plt.show()
    print(f"\n[visualize_results] Figure đã lưu → {save_path}")


# ─────────────────────────────────────────────────────────────────
#  TIỆN ÍCH · HIỆU CHỈNH TỌA ĐỘ (Calibration)
# ─────────────────────────────────────────────────────────────────

def show_calibration_grid(rotated_image: np.ndarray,
                          grid_step: int = 50,
                          save_path: str = 'calibration_grid.png') -> None:
    """
    Vẽ lưới tọa độ lên ảnh đã xoay giúp bạn xác định
    tọa độ (x1, y1, x2, y2) chính xác cho từng ngăn.

    Cách dùng:
        1. Chạy hàm này sau khi đã xoay ảnh.
        2. Mở file calibration_grid.png.
        3. Đọc tọa độ từ giao điểm lưới → điền vào phần CONFIG.

    Args:
        rotated_image (np.ndarray): ảnh BGR đã xoay
        grid_step     (int)       : khoảng cách giữa các đường lưới (pixel)
        save_path     (str)       : đường dẫn lưu ảnh lưới
    """
    h, w = rotated_image.shape[:2]
    grid_img = rotated_image.copy()

    # ── Vẽ đường kẻ dọc (theo X) ──────────────────────────────
    for x in range(0, w, grid_step):
        color = (0, 80, 200) if x % (grid_step * 5) == 0 else (60, 60, 60)
        thick = 2 if x % (grid_step * 5) == 0 else 1
        cv2.line(grid_img, (x, 0), (x, h), color, thick)

    # ── Vẽ đường kẻ ngang (theo Y) ────────────────────────────
    for y in range(0, h, grid_step):
        color = (0, 80, 200) if y % (grid_step * 5) == 0 else (60, 60, 60)
        thick = 2 if y % (grid_step * 5) == 0 else 1
        cv2.line(grid_img, (0, y), (w, y), color, thick)

    # ── Thêm nhãn tọa độ tại các giao điểm lớn (×5) ─────────
    font_scale = max(0.35, min(0.6, w / 1500))
    for x in range(0, w, grid_step * 5):
        for y in range(0, h, grid_step * 5):
            if x == 0 and y == 0:
                continue
            cv2.putText(
                grid_img, f"({x},{y})", (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 200, 255), 1, cv2.LINE_AA,
            )

    # ── Hiển thị và lưu ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB))
    ax.set_title(
        f"Lưới Hiệu Chỉnh Tọa Độ  |  Bước lưới: {grid_step}px"
        f"  |  Kích thước ảnh: {w}×{h}px",
        fontsize=11,
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[calibration] Lưới tọa độ đã lưu → {save_path}")
    print(f"[calibration] Kích thước ảnh xoay: {w}×{h} px")
    print(f"[calibration] Hãy đọc tọa độ từ ảnh và cập nhật các biến:")
    for name in ['SOUP_BOX', 'RICE_BOX', 'COMP1_BOX', 'COMP2_BOX', 'COMP3_BOX']:
        print(f"              {name} = (x1, y1, x2, y2)")


# ─────────────────────────────────────────────────────────────────
#  TIỆN ÍCH · TẠO ẢNH DEMO GIẢ LẬP
# ─────────────────────────────────────────────────────────────────

def create_demo_tray(size: int = 640,
                     rice_at: str = 'top-left') -> np.ndarray:
    """
    Tạo ảnh khay cơm giả lập để kiểm tra pipeline mà không cần ảnh thực.

    Layout khay 5 ngăn:
    ┌──────────────┬──────────────────────────┐
    │              │                          │
    │   góc TL     │         góc TR           │
    │              │                          │
    ├──────────────┼────────────┬─────────────┤
    │              │            │             │
    │   góc BL     │  (giữa)   │   góc BR    │
    │              │            │             │
    └──────────────┴────────────┴─────────────┘

    Args:
        size    (int): kích thước ảnh vuông (pixel)
        rice_at (str): vị trí ngăn cơm trắng — góc nào trong 4 góc

    Returns:
        np.ndarray: ảnh BGR giả lập
    """
    img = np.full((size, size, 3), 55, dtype=np.uint8)   # nền xám sẫm

    # Vẽ mặt khay (nhựa xám nhạt)
    margin = 20
    cv2.rectangle(img,
                  (margin, margin),
                  (size - margin, size - margin),
                  (165, 160, 155), -1)
    cv2.rectangle(img,
                  (margin, margin),
                  (size - margin, size - margin),
                  (90, 85, 80), 4)

    half = size // 2

    # Màu sắc cho từng vị trí (mặc định: không phải cơm)
    food_colors = {
        'top-left':     (140, 180, 90),    # rau xanh
        'top-right':    (60,  100, 180),   # canh xanh
        'bottom-left':  (55,  80,  160),   # thịt đỏ-xanh
        'bottom-right': (35,  100, 180),   # đồ kho
        'center':       (120, 145, 100),   # món trộn
    }
    # Đặt màu trắng (cơm) vào vị trí rice_at
    food_colors[rice_at] = (235, 240, 242)   # gần trắng = cơm

    # Định nghĩa 5 ngăn (x1, y1, x2, y2)
    pad  = margin + 5
    ngans = {
        'top-left':     (pad,         pad,          half - 5,     half - 5    ),
        'top-right':    (half + 5,    pad,          size - pad,   half - 5    ),
        'bottom-left':  (pad,         half + 5,     half - 5,     size - pad  ),
        'bottom-right': (half + 5,    half + 5,     size - pad,   size - pad  ),
        'center':       (half - 60,   half - 60,    half + 60,    half + 60   ),
    }

    for name, (x1, y1, x2, y2) in ngans.items():
        color = food_colors[name]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (40, 38, 35), 2)

        # Thêm texture đơn giản (chấm ngẫu nhiên)
        rng = np.random.default_rng(hash(name) % 2**32)
        n_dots = 80
        xs = rng.integers(x1 + 5, x2 - 5, n_dots)
        ys = rng.integers(y1 + 5, y2 - 5, n_dots)
        for dx, dy in zip(xs, ys):
            noise = int(rng.integers(-30, 30))
            dot_c = tuple(max(0, min(255, c + noise)) for c in color)
            cv2.circle(img, (int(dx), int(dy)), 4, dot_c, -1)

        # Nhãn vị trí
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        text   = 'CƠM' if name == rice_at else name.upper()[:2]
        txt_c  = (20, 20, 20) if name == rice_at else (200, 200, 200)
        cv2.putText(img, text, (cx - 22, cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_c, 2, cv2.LINE_AA)

    # Nhãn tổng thể góc trên ảnh
    info = f"DEMO — COM o: {rice_at}"
    cv2.putText(img, info, (margin + 5, margin - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return img


# ─────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────

def main(use_demo:         bool = True,
         demo_rice_at:     str  = 'top-left',
         image_path:       str  = None,
         run_calibration:  bool = False) -> tuple:
    """
    Chạy toàn bộ pipeline từ đầu đến cuối.

    Args:
        use_demo        : True → dùng ảnh giả lập; False → upload/đọc file
        demo_rice_at    : Vị trí ngăn cơm trong ảnh demo
                          ('top-left' | 'top-right' | 'bottom-left' | 'bottom-right')
        image_path      : Đường dẫn file ảnh (nếu use_demo=False và không upload)
        run_calibration : True → hiện thêm lưới tọa độ để đo tọa độ box

    Returns:
        (rotated_image, crops_dict)
    """
    print("=" * 62)
    print("  PIPELINE NHẬN DIỆN NGĂN KHAY CƠM CĂN TIN")
    print("=" * 62)

    # ── Bước 0: Tải ảnh ──────────────────────────────────────────
    print("\n── Bước 0 · Tải ảnh ─────────────────────────────────────")
    if use_demo:
        print(f"   Chế độ: DEMO  (ngăn cơm đặt ở '{demo_rice_at}')")
        original = create_demo_tray(size=640, rice_at=demo_rice_at)
    else:
        original = load_image(image_path)

    # ── Bước 1: Phát hiện góc ngăn cơm ──────────────────────────
    print("\n── Bước 1 · Phát hiện vị trí ngăn cơm ──────────────────")
    rice_corner = detect_rice_corner(original, debug=True)

    # ── Bước 2: Xoay về hướng chuẩn ─────────────────────────────
    print("\n── Bước 2 · Xoay ảnh về hướng chuẩn ───────────────────")
    rotated = rotate_to_standard(original, rice_corner)

    # ── (Tuỳ chọn) Hiển thị lưới tọa độ để hiệu chỉnh ──────────
    if run_calibration:
        print("\n── Hiệu chỉnh · Lưới tọa độ ────────────────────────────")
        show_calibration_grid(rotated, grid_step=50)

    # ── Bước 3: Crop 5 ngăn ──────────────────────────────────────
    print("\n── Bước 3 · Crop 5 ngăn ────────────────────────────────")
    crops = crop_compartments(rotated)

    # ── Bước 4: Hiển thị tổng hợp ────────────────────────────────
    print("\n── Bước 4 · Hiển thị kết quả ───────────────────────────")
    visualize_results(original, rotated, crops, rice_corner)

    print("\n" + "=" * 62)
    print("  Hoàn tất pipeline ✓")
    print("=" * 62)
    return rotated, crops
