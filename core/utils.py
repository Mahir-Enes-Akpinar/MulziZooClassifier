# core/utils.py
import cv2
import numpy as np
from PIL import Image
import io

try:
    from rembg import remove
    REMBG_AVAILABLE = True
    print("rembg kütüphanesi bulundu. Arkaplan kaldırma etkin. (utils)")
except ImportError:
    REMBG_AVAILABLE = False
    print("rembg kütüphanesi bulunamadı. Arkaplan kaldırma devre dışı. (utils)")
    print("Arkaplan kaldırma için: pip install rembg")

def remove_background_image(image_bytes):
    """Görüntüden arkaplanı kaldırır"""
    if REMBG_AVAILABLE:
        try:
            return remove(image_bytes)
        except Exception as e:
            print(f"Arkaplan kaldırma hatası: {e}")
            raise e
    else:
        raise ImportError("rembg kütüphanesi yüklü değil.")

def apply_gaussian_blur(image_np, kernel_size=(7, 7), sigma=1.5):
    """Gaussian bulanıklaştırma uygular"""
    return cv2.medianBlur(image_np, 5)

def apply_clahe_equalization(image_np, clip_limit=2.0, tile_grid_size=(8, 8)):
    """CLAHE histogram eşitleme uygular"""
    
    r, g, b = cv2.split(image_np)
    r_eq = cv2.equalizeHist(r)
    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)
    return cv2.merge((r_eq, g_eq, b_eq))

def apply_canny_edge_detection(image_np, threshold1, threshold2):
    """Canny kenar algılama uygular"""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_rgb = np.stack((edges,) * 3, axis=-1)
    return Image.fromarray(edges_rgb)