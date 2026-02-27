import cv2
import numpy as np
import matplotlib.pyplot as plt


# =====================================================
# 1️⃣ ANALISIS MODEL WARNA
# =====================================================

def analyze_color_models(image):

    results = {}

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Skin Detection (HSV) ---
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # --- Shadow Removal (LAB - CLAHE) ---
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    L_enhanced = clahe.apply(L)
    lab_enhanced = cv2.merge((L_enhanced, A, B))
    shadow_removed = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # --- Text Extraction (Adaptive Threshold) ---
    text_thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # --- Object Detection (Edge Detection) ---
    edges = cv2.Canny(gray, 100, 200)

    
    edges_colored = np.zeros_like(image)
    edges_colored[:,:,2] = edges  # channel merah

    results["skin_mask"] = skin_mask
    results["shadow_removed"] = shadow_removed
    results["text_thresh"] = text_thresh
    results["edges"] = edges_colored

    # Analisis sederhana
    results["analysis"] = {
        "skin_pixels": np.sum(skin_mask > 0),
        "contrast_before": np.std(L),
        "contrast_after": np.std(L_enhanced),
        "edge_pixels": np.sum(edges > 0)
    }

    return results


# =====================================================
# 2️⃣ SIMULASI ALIASING
# =====================================================

def simulate_aliasing(image, factors):

    results = {}

    h, w = image.shape[:2]

    for factor in factors:

        new_w = w // factor
        new_h = h // factor

        # Tanpa anti-aliasing
        down_nearest = cv2.resize(image, (new_w, new_h),
                                  interpolation=cv2.INTER_NEAREST)
        restored_nearest = cv2.resize(down_nearest, (w, h),
                                      interpolation=cv2.INTER_NEAREST)

        # Dengan anti-aliasing
        down_area = cv2.resize(image, (new_w, new_h),
                               interpolation=cv2.INTER_AREA)
        restored_area = cv2.resize(down_area, (w, h),
                                   interpolation=cv2.INTER_LINEAR)

        mse_nearest = np.mean((image - restored_nearest) ** 2)
        mse_area = np.mean((image - restored_area) ** 2)

        results[factor] = {
            "nearest": restored_nearest,
            "area": restored_area,
            "mse_nearest": mse_nearest,
            "mse_area": mse_area
        }

    return results


# =====================================================
# 3️⃣ MAIN PROGRAM
# =====================================================

if __name__ == "__main__":

    
    img = cv2.imread("Amerika.jpeg")

    if img is None:
        print("Gambar tidak ditemukan!")
        exit()

    # =========================
    # ANALISIS MODEL WARNA
    # =========================
    color_results = analyze_color_models(img)

    print("===== ANALISIS MODEL WARNA =====")
    for key, value in color_results["analysis"].items():
        print(f"{key} : {value}")
    print()

    # =========================
    # SIMULASI ALIASING
    # =========================
    factors = [2, 4, 8]
    aliasing_results = simulate_aliasing(img, factors)

    print("===== ANALISIS ALIASING =====")
    for factor in factors:
        print(f"Factor {factor}")
        print("MSE Nearest :", aliasing_results[factor]["mse_nearest"])
        print("MSE Area    :", aliasing_results[factor]["mse_area"])
        print()

    # =========================
    # VISUALISASI
    # =========================
    plt.figure(figsize=(16,10))

    # Baris 1 → Original & Model Warna
    plt.subplot(3,4,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(3,4,2)
    plt.imshow(color_results["skin_mask"], cmap="gray")
    plt.title("Skin Detection (HSV)")
    plt.axis("off")

    plt.subplot(3,4,3)
    plt.imshow(cv2.cvtColor(color_results["shadow_removed"], cv2.COLOR_BGR2RGB))
    plt.title("Shadow Removal (LAB)")
    plt.axis("off")

    plt.subplot(3,4,4)
    plt.imshow(color_results["edges"])
    plt.title("Object Detection (Edges)")
    plt.axis("off")

    # Baris 2 & 3 → Aliasing
    index = 5
    for factor in factors:
        plt.subplot(3,4,index)
        plt.imshow(cv2.cvtColor(aliasing_results[factor]["nearest"], cv2.COLOR_BGR2RGB))
        plt.title(f"Nearest x{factor}")
        plt.axis("off")
        index += 1

    for factor in factors:
        plt.subplot(3,4,index)
        plt.imshow(cv2.cvtColor(aliasing_results[factor]["area"], cv2.COLOR_BGR2RGB))
        plt.title(f"Area x{factor}")
        plt.axis("off")
        index += 1

    plt.tight_layout()
    plt.show()