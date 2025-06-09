# core/prediction_thread.py
import numpy as np
import cv2
import torch
from PIL import Image
import io
from PyQt5.QtCore import QThread, pyqtSignal
from core.utils import apply_gaussian_blur, apply_clahe_equalization, apply_canny_edge_detection

# rembg kütüphanesini kontrol et (bu blok burada kalmalı)
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("rembg kütüphanesi bulunamadı. Arkaplan kaldırma devre dışı.")

class PredictionThread(QThread):
    prediction_complete = pyqtSignal(str, float, list, list, object, object) # Ek olarak processed_img ve edge_img döndürüyoruz
    progress_update = pyqtSignal(int)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, model, image_path, transform, edge_transform, device, class_names, preprocess_options):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.transform = transform
        self.edge_transform = edge_transform
        self.device = device
        self.class_names = class_names
        self.preprocess_options = preprocess_options
    
    def run(self):
        try:
            self.progress_update.emit(10)
            img_original_pil = Image.open(self.image_path).convert('RGB')
            img_original_np = np.array(img_original_pil)

            # --- Modelin birinci girişi için ön işleme (Blur + Histogram Eşitleme) ---
            img_for_model1_np = img_original_np.copy()
            
            processed_img_pil = Image.fromarray(img_for_model1_np.astype('uint8'))
            img_tensor = self.transform(processed_img_pil).unsqueeze(0).to(self.device)

            self.progress_update.emit(40)

            # --- Modelin ikinci girişi için ön işleme (Arkaplan Kaldırma + Kenar Algılama) ---
            img_for_model2_np = img_original_np.copy()

            if self.preprocess_options.get('remove_background', False) and REMBG_AVAILABLE:
                self.progress_update.emit(50)
                try:
                    with open(self.image_path, "rb") as f:
                        input_bytes = f.read()
                    output_bytes = remove(input_bytes)
                    no_bg_img = Image.open(io.BytesIO(output_bytes)).convert('RGB')
                    img_for_model2_np = np.array(no_bg_img)
                except Exception as e:
                    print(f"Arkaplan kaldırma hatası (tahmin): {e}")
                    # Arkaplan kaldırma hatası durumunda orijinali kullanmaya devam edebiliriz
                    # veya kenar algılamanın başarısız olması durumunda siyah bir resim verebiliriz.
                    
            if self.preprocess_options.get('equalize_hist', False):
                
                r, g, b = cv2.split(img_for_model2_np)
                r_eq = cv2.equalizeHist(r)
                g_eq = cv2.equalizeHist(g)
                b_eq = cv2.equalizeHist(b)
                img_for_model2_np = cv2.merge((r_eq, g_eq, b_eq))

            if self.preprocess_options.get('reduce_noise', False):
                img_for_model2_np = cv2.medianBlur(img_for_model2_np, 5)
            self.progress_update.emit(60)

            # Kenar Algılama
            try:
                gray = cv2.cvtColor(img_for_model2_np, cv2.COLOR_RGB2GRAY)
                threshold1 = self.preprocess_options.get('canny_threshold1', 100)
                threshold2 = self.preprocess_options.get('canny_threshold2', 200)
                edges = cv2.Canny(gray, threshold1, threshold2)
                edges_rgb = np.stack((edges,) * 3, axis=-1)
                edge_img_pil = Image.fromarray(edges_rgb)
            except Exception as e:
                print(f"Kenar algılama hatası (tahmin): {e}")
                edge_img_pil = Image.new('RGB', img_original_pil.size, color='black') # Hata durumunda siyah resim
            
            edge_tensor = self.edge_transform(edge_img_pil).unsqueeze(0).to(self.device)

            self.model.eval()
            self.progress_update.emit(70)
            with torch.no_grad():
                outputs = self.model(img_tensor, edge_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probs, 1)
            
            predicted_class = self.class_names[pred_idx.item()]
            confidence = conf.item()
            probs = probs.cpu().numpy()[0]
            
            top5_idx = np.argsort(probs)[-5:][::-1]
            top5_classes = [self.class_names[i] for i in top5_idx]
            top5_probs = probs[top5_idx].tolist()
            
            self.progress_update.emit(90)
            
            # Tahmin tamamlandığında, işlenmiş ve kenar görüntülerini de geri gönder
            self.prediction_complete.emit(
                predicted_class, confidence, top5_classes, top5_probs,
                processed_img_pil, edge_img_pil # UI için güncellenmiş görüntüler
            )
            self.progress_update.emit(100)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
            self.progress_update.emit(0)