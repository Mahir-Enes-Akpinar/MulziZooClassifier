# ui/main_window.py
import os
import zipfile
from PyQt5.QtWidgets import (QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, 
                             QWidget, QFileDialog, QProgressBar, QMessageBox, QFrame, 
                             QSplitter, QCheckBox, QGridLayout, QSlider, QStatusBar,
                             QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt, QSize, QTimer, QThread, pyqtSignal
import numpy as np
import io
import json
from pathlib import Path

from ui.widgets import ImagePreviewLabel, ResultCanvas, TestResultsWidget
from core.model import load_multizoo_model
import torch 
from PIL import Image
from core.utils import apply_gaussian_blur, apply_clahe_equalization, apply_canny_edge_detection, remove_background_image

# rembg kütüphanesini kontrol et
try:
    from rembg import remove
    REMBG_AVAILABLE = True
    print("rembg kütüphanesi bulundu. Arkaplan kaldırma etkin. (main_window)")
except ImportError:
    REMBG_AVAILABLE = False
    print("rembg kütüphanesi bulunamadı. Arkaplan kaldırma devre dışı. (main_window)")
    print("Arkaplan kaldırma için: pip install rembg")

# PredictionThread import'unu dinamik yap
from core.prediction_thread import PredictionThread

class TestFileThread(QThread):
    """Test dosyası işleme thread'i"""
    test_complete = pyqtSignal(list)  # Test sonuçları listesi
    progress_update = pyqtSignal(int, str)  # Progress ve mesaj
    error_occurred = pyqtSignal(str)
    
    def __init__(self, model, test_path, transform, edge_transform, device, class_names, preprocess_options):
        super().__init__()
        self.model = model
        self.test_path = test_path
        self.transform = transform
        self.edge_transform = edge_transform
        self.device = device
        self.class_names = class_names
        self.preprocess_options = preprocess_options
        
    def extract_true_label_from_filename(self, filename, filepath):
        """Dosya adından veya yolundan gerçek etiketi çıkar"""
        # Önce dosya yolundan klasör adını kontrol et
        parent_dir = os.path.basename(os.path.dirname(filepath))
        if parent_dir.lower() in [cls.lower() for cls in self.class_names]:
            # Klasör adı sınıf adıyla eşleşiyor
            for cls in self.class_names:
                if cls.lower() == parent_dir.lower():
                    return cls
        
        # Dosya adından sınıf adını çıkarmaya çalış
        filename_lower = filename.lower()
        filename_base = os.path.splitext(filename)[0].lower()  # Uzantı olmadan
        
        # Sınıf adlarını dosya adında ara
        for cls in self.class_names:
            cls_lower = cls.lower()
            # Tam eşleşme kontrolü (kelime sınırları ile)
            if cls_lower in filename_lower:
                # Kelimenin başında/sonunda olduğunu kontrol et
                import re
                pattern = r'\b' + re.escape(cls_lower) + r'\b'
                if re.search(pattern, filename_lower):
                    return cls
        
        # Eğer bulamazsa None döndür (bilinmeyen)
        return None
        
    def run(self):
        try:
            # Test dosyalarını topla
            image_files = []
            test_path = Path(self.test_path)
            
            if test_path.is_file() and test_path.suffix.lower() == '.zip':
                # ZIP dosyası durumu
                with zipfile.ZipFile(test_path, 'r') as zip_ref:
                    temp_dir = test_path.parent / 'temp_test'
                    temp_dir.mkdir(exist_ok=True)
                    zip_ref.extractall(temp_dir)
                    
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                full_path = os.path.join(root, file)
                                image_files.append((full_path, file))
            elif test_path.is_dir():
                # Klasör durumu
                for root, dirs, files in os.walk(test_path):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            full_path = os.path.join(root, file)
                            image_files.append((full_path, file))
            else:
                raise ValueError("Geçersiz test dosyası formatı")
            
            if not image_files:
                raise ValueError("Test dosyasında görüntü bulunamadı")
            
            results = []
            total_files = len(image_files)
            
            for i, (image_path, filename) in enumerate(image_files):
                try:
                    # Progress güncelle
                    progress = int((i / total_files) * 100)
                    self.progress_update.emit(progress, f"İşleniyor: {filename}")
                    
                    # Gerçek etiketi çıkar
                    true_label = self.extract_true_label_from_filename(filename, image_path)
                    
                    # Görüntüyü yükle ve işle
                    result = self.process_single_image(image_path, filename)
                    
                    # Doğruluk kontrolü ekle
                    result['true_label'] = true_label
                    result['has_ground_truth'] = true_label is not None
                    
                    if true_label is not None:
                        result['is_correct'] = result['predicted_class'].lower() == true_label.lower()
                        result['accuracy_status'] = 'Doğru' if result['is_correct'] else 'Yanlış'
                    else:
                        result['is_correct'] = None
                        result['accuracy_status'] = 'Bilinmiyor'
                    
                    results.append(result)
                    
                except Exception as e:
                    # Tek görüntü hatası - devam et
                    results.append({
                        'filename': filename,
                        'predicted_class': 'HATA',
                        'confidence': 0.0,
                        'top5_classes': [],
                        'top5_probs': [],
                        'true_label': None,
                        'has_ground_truth': False,
                        'is_correct': None,
                        'accuracy_status': 'Hata',
                        'error': str(e)
                    })
            
            self.progress_update.emit(100, "Test tamamlandı")
            self.test_complete.emit(results)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def process_single_image(self, image_path, filename):
        """Tek bir görüntüyü işle"""
        img_original_pil = Image.open(image_path).convert('RGB')
        img_original_np = np.array(img_original_pil)

        # --- Modelin birinci girişi için ön işleme ---
        img_for_model1_np = img_original_np.copy()

        processed_img_pil = Image.fromarray(img_for_model1_np.astype('uint8'))
        img_tensor = self.transform(processed_img_pil).unsqueeze(0).to(self.device)

        # --- Modelin ikinci girişi için ön işleme ---
        img_for_model2_np = img_original_np.copy()

        if self.preprocess_options.get('remove_background', False) and REMBG_AVAILABLE:
            try:
                with open(image_path, "rb") as f:
                    input_bytes = f.read()
                output_bytes = remove_background_image(input_bytes)
                no_bg_img = Image.open(io.BytesIO(output_bytes)).convert('RGB')
                img_for_model2_np = np.array(no_bg_img)
            except Exception as e:
                print(f"Arkaplan kaldırma hatası: {e}")
                  
        if self.preprocess_options.get('equalize_hist', False):
            img_for_model2_np = apply_clahe_equalization(img_for_model2_np)
            
        if self.preprocess_options.get('reduce_noise', False):
            img_for_model2_np = apply_gaussian_blur(img_for_model2_np)
        
        # Kenar Algılama
        try:
            threshold1 = self.preprocess_options.get('canny_threshold1', 100)
            threshold2 = self.preprocess_options.get('canny_threshold2', 200)
            edge_img_pil = apply_canny_edge_detection(img_for_model2_np, threshold1, threshold2)
        except Exception as e:
            print(f"Kenar algılama hatası: {e}")
            edge_img_pil = Image.new('RGB', img_original_pil.size, color='black')
        
        edge_tensor = self.edge_transform(edge_img_pil).unsqueeze(0).to(self.device)

        # Model tahmini
        self.model.eval()
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
        
        return {
            'filename': filename,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top5_classes': top5_classes,
            'top5_probs': top5_probs
        }

class MultiZooApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MultiZoo Hayvan Sınıflandırıcı')
        self.setGeometry(100, 100, 1400, 900)
        self.setWindowIcon(QIcon('assets/icon.png')) 
        self.model_loaded = False
        self.debug_mode = True 
        
        self.preprocess_options = {
            'remove_background': True,
            'equalize_hist': True,
            'reduce_noise': True,
            'canny_threshold1': 100,
            'canny_threshold2': 200
        }
        
        self.init_ui()
        QTimer.singleShot(500, self.load_model_dialog)
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Tab Widget ekle
        self.tab_widget = QTabWidget()
        
        # Tek görüntü tab'ı
        single_image_tab = self.create_single_image_tab()
        self.tab_widget.addTab(single_image_tab, "Tek Görüntü")
        
        # Test dosyası tab'ı
        test_file_tab = self.create_test_file_tab()
        self.tab_widget.addTab(test_file_tab, "Test Dosyası")
        
        main_layout.addWidget(self.tab_widget)
        
        # Progress bar (tüm tab'lar için ortak)
        progress_frame = QFrame()
        progress_frame.setFrameShape(QFrame.StyledPanel)
        progress_frame.setFrameShadow(QFrame.Raised)
        progress_layout = QVBoxLayout(progress_frame)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        progress_layout.addWidget(self.progress_bar)
        main_layout.addWidget(progress_frame)
        
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('Başlamak için bir model yükleyin')

    def create_single_image_tab(self):
        """Tek görüntü işleme tab'ını oluştur"""
        tab_widget = QWidget()
        main_layout = QVBoxLayout(tab_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setChildrenCollapsible(False)
        
        # Left Panel (Images)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        img_control_layout = QHBoxLayout()
        
        self.select_button = QPushButton('Görüntü Seç')
        self.select_button.setStyleSheet("background-color: #3498db; color: white;")
        self.select_button.setIcon(QIcon.fromTheme("document-open"))
        self.select_button.setIconSize(QSize(16, 16))
        self.select_button.clicked.connect(self.select_image)
        self.select_button.setEnabled(False)
        img_control_layout.addWidget(self.select_button)
        
        self.preprocess_button = QPushButton('Ön İşleme Ayarları')
        self.preprocess_button.setStyleSheet("background-color: #9b59b6; color: white;")
        self.preprocess_button.clicked.connect(self.show_preprocess_settings)
        self.preprocess_button.setEnabled(False)
        img_control_layout.addWidget(self.preprocess_button)
        
        left_layout.addLayout(img_control_layout)
        
        img_grid = QGridLayout()
        orig_title = QLabel('Orijinal Görüntü')
        orig_title.setAlignment(Qt.AlignCenter)
        orig_title.setFont(QFont('Arial', 11, QFont.Bold))
        img_grid.addWidget(orig_title, 0, 0)
        self.orig_img_label = ImagePreviewLabel()
        img_grid.addWidget(self.orig_img_label, 1, 0)
        
        processed_title = QLabel('İşlenmiş Görüntü')
        processed_title.setAlignment(Qt.AlignCenter)
        processed_title.setFont(QFont('Arial', 11, QFont.Bold))
        img_grid.addWidget(processed_title, 0, 1)
        self.processed_img_label = ImagePreviewLabel()
        img_grid.addWidget(self.processed_img_label, 1, 1)
        
        edge_title = QLabel('Kenar Görüntüsü')
        edge_title.setAlignment(Qt.AlignCenter)
        edge_title.setFont(QFont('Arial', 11, QFont.Bold))
        img_grid.addWidget(edge_title, 2, 0, 1, 2)
        self.edge_img_label = ImagePreviewLabel()
        img_grid.addWidget(self.edge_img_label, 3, 0, 1, 2)
        
        left_layout.addLayout(img_grid)
        
        # Right Panel (Results)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        result_title = QLabel('Sınıflandırma Sonucu')
        result_title.setAlignment(Qt.AlignCenter)
        result_title.setFont(QFont('Arial', 14, QFont.Bold))
        result_title.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        right_layout.addWidget(result_title)
        
        self.prediction_label = QLabel('Bekleniyor...')
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setFont(QFont('Arial', 18, QFont.Bold))
        self.prediction_label.setStyleSheet("""
            color: #2c3e50; background-color: #ecf0f1; border-radius: 5px;
            padding: 10px; margin: 10px;
        """)
        self.prediction_label.setWordWrap(True)
        right_layout.addWidget(self.prediction_label)
        
        self.confidence_label = QLabel('Güven: -')
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setFont(QFont('Arial', 12))
        self.confidence_label.setStyleSheet("color: #7f8c8d; margin-bottom: 15px;")
        right_layout.addWidget(self.confidence_label)
        
        result_frame = QFrame()
        result_frame.setFrameShape(QFrame.StyledPanel)
        result_frame.setFrameShadow(QFrame.Raised)
        result_frame.setStyleSheet("background-color: #ffffff;")
        result_layout = QVBoxLayout(result_frame)
        self.result_canvas = ResultCanvas(result_frame, width=5, height=4, dpi=100)
        result_layout.addWidget(self.result_canvas)
        right_layout.addWidget(result_frame)
        
        self.predict_button = QPushButton('Sınıflandır')
        self.predict_button.setMinimumHeight(40)
        self.predict_button.setStyleSheet("background-color: #2ecc71; color: white; font-size: 14px;")
        self.predict_button.setIcon(QIcon.fromTheme("system-run"))
        self.predict_button.setIconSize(QSize(18, 18))
        self.predict_button.clicked.connect(self.predict_image)
        self.predict_button.setEnabled(False)
        right_layout.addWidget(self.predict_button)
        
        self.load_model_button = QPushButton('Model Yükle/Değiştir')
        self.load_model_button.setStyleSheet("background-color: #e74c3c; color: white;")
        self.load_model_button.clicked.connect(self.load_model_dialog)
        right_layout.addWidget(self.load_model_button)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([500, 500])
        main_layout.addWidget(splitter)
        
        return tab_widget

    def create_test_file_tab(self):
        """Test dosyası işleme tab'ını oluştur"""
        tab_widget = QWidget()
        main_layout = QVBoxLayout(tab_widget)
        
        # Test dosyası yükleme kontrolleri
        control_layout = QHBoxLayout()
        
        self.load_test_button = QPushButton('Test Dosyası Yükle')
        self.load_test_button.setStyleSheet("background-color: #f39c12; color: white; font-weight: bold;")
        self.load_test_button.setIcon(QIcon.fromTheme("folder-open"))
        self.load_test_button.setIconSize(QSize(16, 16))
        self.load_test_button.clicked.connect(self.load_test_file)
        self.load_test_button.setEnabled(False)
        control_layout.addWidget(self.load_test_button)
        

        
        self.export_results_button = QPushButton('Sonuçları Dışa Aktar')
        self.export_results_button.setStyleSheet("background-color: #27ae60; color: white;")
        self.export_results_button.setIcon(QIcon.fromTheme("document-save"))
        self.export_results_button.setIconSize(QSize(16, 16))
        self.export_results_button.clicked.connect(self.export_test_results)
        self.export_results_button.setEnabled(False)
        control_layout.addWidget(self.export_results_button)
        
        control_layout.addStretch()
        main_layout.addLayout(control_layout)
        
        # Test sonuçları widget'ı
        self.test_results_widget = TestResultsWidget()
        main_layout.addWidget(self.test_results_widget)
        
        # Info label
        info_label = QLabel("Test dosyası olarak ZIP dosyası veya görüntülerin bulunduğu klasörü seçebilirsiniz.")
        info_label.setStyleSheet("color: #7f8c8d; font-style: italic; margin: 5px;")
        main_layout.addWidget(info_label)
        
        return tab_widget

    def load_test_file(self):
        """Test dosyası yükle"""
        # Önce kullanıcıya seçenek sun
        choice = QMessageBox.question(
            self, 
            'Test Dosyası Türü', 
            'Test dosyası türünü seçin:\n\n• Klasör: Görüntülerin bulunduğu klasörü seç\n• ZIP: ZIP dosyasını seç',
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes
        )
        
        if choice == QMessageBox.Cancel:
            return
        elif choice == QMessageBox.Yes:
            # Klasör seç
            test_path = QFileDialog.getExistingDirectory(
                self, "Test Klasörünü Seç", ""
            )
        else:
            # ZIP dosyası seç
            test_path, _ = QFileDialog.getOpenFileName(
                self, "Test ZIP Dosyasını Seç", "", "ZIP Dosyaları (*.zip)"
            )
        
        if test_path:
            self.run_test_file(test_path)

    def run_test_file(self, test_path):
        """Test dosyasını çalıştır"""
        if not self.model_loaded:
            QMessageBox.warning(self, "Uyarı", "Önce bir model yüklemelisiniz!")
            return
        
        self.load_test_button.setEnabled(False)
        self.export_results_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.statusBar.showMessage(f"Test dosyası işleniyor: {os.path.basename(test_path)}")
        
        # Test results widget'ını temizle
        self.test_results_widget.clear_results()
        
        # Test thread'ini başlat
        self.test_thread = TestFileThread(
            self.model, test_path, self.transform, 
            self.edge_transform, self.device, self.class_names,
            self.preprocess_options
        )
        
        self.test_thread.test_complete.connect(self.update_test_results)
        self.test_thread.progress_update.connect(self.update_test_progress)
        self.test_thread.error_occurred.connect(self.handle_test_error)
        self.test_thread.start()

    def update_test_progress(self, progress, message):
        """Test ilerlemesini güncelle"""
        self.progress_bar.setValue(progress)
        self.statusBar.showMessage(message)

    def update_test_results(self, results):
        """Test sonuçlarını güncelle"""
        self.test_results = results
        self.test_results_widget.update_results(results)
        self.load_test_button.setEnabled(True)
        self.export_results_button.setEnabled(True)
        
        # İstatistikleri hesapla
        total = len(results)
        errors = sum(1 for r in results if 'error' in r)
        success = total - errors
        
        # Doğruluk istatistikleri
        correct_predictions = sum(1 for r in results if r.get('is_correct') == True)
        incorrect_predictions = sum(1 for r in results if r.get('is_correct') == False)
        
        if success > 0:
            avg_confidence = np.mean([r['confidence'] for r in results if 'error' not in r])
            
            status_parts = [
                f"Test tamamlandı - Toplam: {total}",
                f"Başarılı: {success}",
                f"Hata: {errors}"
            ]
            
            if correct_predictions + incorrect_predictions > 0:
                accuracy = (correct_predictions / (correct_predictions + incorrect_predictions)) * 100
                status_parts.extend([
                    f"Test Accuracy: {accuracy:.2f}%",
                    f"Ortalama Güven: {avg_confidence:.4f}"
                ])
            else:
                status_parts.append(f"Ortalama Güven: {avg_confidence:.4f}")
            
            status_message = " | ".join(status_parts)
        else:
            status_message = f"Test tamamlandı - Toplam: {total} | Tüm dosyalarda hata oluştu"
        
        self.statusBar.showMessage(status_message)

    def handle_test_error(self, error_message):
        """Test hatasını işle"""
        QMessageBox.critical(self, "Test Hatası", f"Test sırasında hata oluştu: {error_message}")
        self.load_test_button.setEnabled(True)
        self.statusBar.showMessage("Test hatası oluştu")

    def export_test_results(self):
        """Test sonuçlarını dışa aktar"""
        if not hasattr(self, 'test_results') or not self.test_results:
            QMessageBox.warning(self, "Uyarı", "Dışa aktarılacak test sonucu bulunamadı!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Test Sonuçlarını Kaydet", "test_results.json", 
            "JSON Dosyaları (*.json);;CSV Dosyaları (*.csv)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(self.test_results, f, indent=2, ensure_ascii=False)
                elif file_path.endswith('.csv'):
                    import csv
                    with open(file_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            'Dosya_Adı', 'Cevap', 'Tahmin', 'Güvenirlik', 'Sonuç', 
                            'Top5_Sınıflar', 'Top5_Olasılıklar', 'Hata'
                        ])
                        for result in self.test_results:
                            writer.writerow([
                                result['filename'],
                                result.get('true_label', 'Bilinmiyor'),
                                result['predicted_class'],
                                result['confidence'],
                                result.get('accuracy_status', 'Bilinmiyor'),
                                '|'.join(result.get('top5_classes', [])),
                                '|'.join([str(p) for p in result.get('top5_probs', [])]),
                                result.get('error', '')
                            ])
                
                QMessageBox.information(self, "Başarılı", f"Test sonuçları kaydedildi: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Sonuçlar kaydedilirken hata oluştu: {str(e)}")

    def show_preprocess_settings(self):
        # Bu fonksiyon aynı kalacak - çok uzun olduğu için kısaltıyorum
        settings_dialog = QMessageBox(self)
        settings_dialog.setWindowTitle("Ön İşleme Ayarları")
        settings_dialog.setIcon(QMessageBox.Information)
        
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        
        bg_checkbox = QCheckBox("Arkaplan Kaldırma")
        if REMBG_AVAILABLE:
            bg_checkbox.setChecked(self.preprocess_options.get('remove_background', True))
            bg_checkbox.stateChanged.connect(
                lambda state: self.preprocess_options.update({'remove_background': bool(state)})
            )
            settings_layout.addWidget(bg_checkbox)
        else:
            bg_checkbox.setChecked(False)
            bg_checkbox.setEnabled(False)
            settings_layout.addWidget(bg_checkbox)
            bg_label = QLabel("Arkaplan kaldırma için 'rembg' kütüphanesi gerekiyor.")
            bg_label.setStyleSheet("color: #e74c3c;")
            settings_layout.addWidget(bg_label)
        
        hist_checkbox = QCheckBox("Histogram Eşitleme")
        hist_checkbox.setChecked(self.preprocess_options.get('equalize_hist', True))
        hist_checkbox.stateChanged.connect(
            lambda state: self.preprocess_options.update({'equalize_hist': bool(state)})
        )
        settings_layout.addWidget(hist_checkbox)
        
        noise_checkbox = QCheckBox("Gürültü Azaltma")
        noise_checkbox.setChecked(self.preprocess_options.get('reduce_noise', True))
        noise_checkbox.stateChanged.connect(
            lambda state: self.preprocess_options.update({'reduce_noise': bool(state)})
        )
        settings_layout.addWidget(noise_checkbox)
        
        settings_dialog.setLayout(settings_layout)
        settings_dialog.exec_()
        
        if hasattr(self, 'image_path') and self.image_path:
            self.process_and_display_image(self.image_path)
            
    def reset_preprocess_options(self):
        self.preprocess_options = {
            'remove_background': True, 'equalize_hist': True, 'reduce_noise': True,
            'canny_threshold1': 100, 'canny_threshold2': 200
        }
        QMessageBox.information(self, "Bilgi", "Varsayılan ayarlar geri yüklendi.")
    
    def load_model_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Model Dosyasını Seç", "", "PyTorch Model Dosyaları (*.pth)"
        )
        if file_path:
            self.statusBar.showMessage(f"Model yükleniyor: {os.path.basename(file_path)}")
            self.load_model(file_path)

    def load_model(self, model_path):
        try:
            self.progress_bar.setValue(10)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model, self.class_names, self.transform, self.edge_transform, val_acc, val_f1 = \
                load_multizoo_model(model_path, self.device)
            
            self.model_loaded = True
            self.select_button.setEnabled(True)
            self.preprocess_button.setEnabled(True)
            self.load_test_button.setEnabled(True)
            self.progress_bar.setValue(100)
            
            success_msg = (f"Model başarıyla yüklendi!\n\n"
                        f"Sınıf sayısı: {len(self.class_names)}\n")
            if val_acc > 0: success_msg += f"Doğrulama doğruluğu: {val_acc:.4f}\n"
            if val_f1 > 0: success_msg += f"F1 skoru: {val_f1:.4f}\n"
            success_msg += f"\nCihaz: {self.device}"
            
            QMessageBox.information(self, "Model Yüklendi", success_msg)
            self.statusBar.showMessage(f"Model yüklendi - {len(self.class_names)} sınıf")
            
        except Exception as e:
            error_msg = f"Model yüklenirken hata oluştu: {str(e)}"
            QMessageBox.critical(self, "Hata", error_msg)
            self.statusBar.showMessage("Model yüklenemedi")
            self.progress_bar.setValue(0)
    
    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Görüntü Seç", "", "Görüntü Dosyaları (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.image_path = file_path
            self.process_and_display_image(file_path)
    
    def process_and_display_image(self, file_path):
        try:
            self.progress_bar.setValue(10)
            self.statusBar.showMessage(f"Görüntü işleniyor: {os.path.basename(file_path)}")
            self.display_original_image(file_path)

            # Orijinal görüntüyü yükle
            img_original_pil = Image.open(file_path).convert('RGB')
            img_original_np = np.array(img_original_pil)
            
            # --- processed_img_label için ön işleme (Blur + Histogram Eşitleme) ---
            processed_img_for_display_np = img_original_np.copy()
            
            
            # processed_img_label'da gösterecek görüntüyü ayarla
            processed_img_display_pil = Image.fromarray(processed_img_for_display_np.astype('uint8'))
            self.display_processed_image(processed_img_display_pil)
            
            self.progress_bar.setValue(50)

            # --- edge_img_label için ön işleme (Arkaplan Kaldırma + Kenar Algılama) ---
            edge_base_img_np = img_original_np.copy()

            if self.preprocess_options.get('remove_background', True) and REMBG_AVAILABLE:
                self.statusBar.showMessage("Arkaplan kaldırılıyor (kenar için)...")
                try:
                    with open(file_path, "rb") as f:
                        input_bytes = f.read()
                    output_bytes = remove_background_image(input_bytes)
                    no_bg_img = Image.open(io.BytesIO(output_bytes)).convert('RGB')
                    edge_base_img_np = np.array(no_bg_img)
                except Exception as e:
                    print(f"Arkaplan kaldırma hatası (kenar için): {e}")
                    print("Arkaplan kaldırma başarısız - kenar için orijinal görüntü kullanılıyor.")
                    
            if self.preprocess_options.get('equalize_hist', False):
                
                self.statusBar.showMessage("Histogram eşitleme uygulanıyor (önizleme)...")
                try:
                    edge_base_img_np = apply_clahe_equalization(edge_base_img_np)
                except Exception as e:
                    print(f"Histogram eşitleme hatası (önizleme): {e}")
                    
            if self.preprocess_options.get('reduce_noise', False):
                self.statusBar.showMessage("Gürültü azaltılıyor (önizleme)...")
                try:
                    edge_base_img_np = apply_gaussian_blur(edge_base_img_np)
                except Exception as e:
                    print(f"Gürültü azaltma hatası (önizleme): {e}")
            self.progress_bar.setValue(70)

            # Kenar Algılama
            try:
                self.statusBar.showMessage("Kenar algılama uygulanıyor...")
                thresh1 = self.preprocess_options.get('canny_threshold1', 100)
                thresh2 = self.preprocess_options.get('canny_threshold2', 200)
                edge_img_display_pil = apply_canny_edge_detection(edge_base_img_np, thresh1, thresh2)
            except Exception as e:
                print(f"Kenar algılama hatası: {e}")
                edge_img_display_pil = Image.new('RGB', (224, 224), color='black')
            
            self.display_edge_image(edge_img_display_pil)
            
            self.progress_bar.setValue(90)
            self.predict_button.setEnabled(True)
            self.prediction_label.setText("Bekleniyor...")
            self.confidence_label.setText("Güven: -")
            self.result_canvas.axes.clear()
            self.result_canvas.draw()
            
            self.progress_bar.setValue(100)
            self.statusBar.showMessage(f"Görüntü yüklendi ve işlendi: {os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Görüntü işlenirken hata oluştu: {str(e)}")
            self.statusBar.showMessage("Görüntü işlenemedi")
            self.progress_bar.setValue(0)
    
    def display_original_image(self, file_path):
        pixmap = QPixmap(file_path)
        self.orig_img_label.setPixmap(pixmap)
    
    def display_processed_image(self, img):
        if isinstance(img, Image.Image):
            img_data = img.tobytes("raw", "RGB")
            qimg = QImage(img_data, img.width, img.height, img.width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.processed_img_label.setPixmap(pixmap)
        else:
            pil_img = Image.fromarray(img)
            img_data = pil_img.tobytes("raw", "RGB")
            qimg = QImage(img_data, pil_img.width, pil_img.height, pil_img.width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.processed_img_label.setPixmap(pixmap)
    
    def display_edge_image(self, img):
        if isinstance(img, Image.Image):
            img_data = img.tobytes("raw", "RGB")
            qimg = QImage(img_data, img.width, img.height, img.width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.edge_img_label.setPixmap(pixmap)
        else:
            pil_img = Image.fromarray(img)
            img_data = pil_img.tobytes("raw", "RGB")
            qimg = QImage(img_data, pil_img.width, pil_img.height, pil_img.width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.edge_img_label.setPixmap(pixmap)
    
    def predict_image(self):
        if hasattr(self, 'image_path') and self.model_loaded:
            self.select_button.setEnabled(False)
            self.predict_button.setEnabled(False)
            self.progress_bar.setValue(0)
            self.statusBar.showMessage("Tahmin yapılıyor...")
            
            self.prediction_thread = PredictionThread(
                self.model, self.image_path, self.transform, 
                self.edge_transform, self.device, self.class_names,
                self.preprocess_options
            )
            
            self.prediction_thread.prediction_complete.connect(self.update_result)
            self.prediction_thread.progress_update.connect(self.progress_bar.setValue)
            self.prediction_thread.error_occurred.connect(self.handle_error)
            self.prediction_thread.start()
    
    def update_result(self, predicted_class, confidence, top5_classes, top5_probs, 
                      processed_img, edge_img):
        self.prediction_label.setText(predicted_class)
        confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
        self.confidence_label.setText(f"Güven: {confidence:.4f}")
        self.confidence_label.setStyleSheet(f"color: {confidence_color}; font-weight: bold;")
        self.result_canvas.update_chart(top5_classes, top5_probs)
        self.display_processed_image(processed_img)
        self.display_edge_image(edge_img)
        self.select_button.setEnabled(True)
        self.predict_button.setEnabled(True)
        self.statusBar.showMessage(f"Tahmin: {predicted_class} (Güven: {confidence:.4f})")
    
    def handle_error(self, error_message):
        QMessageBox.critical(self, "Hata", f"Tahmin sırasında hata oluştu: {error_message}")
        self.select_button.setEnabled(True)
        self.predict_button.setEnabled(True)
        self.statusBar.showMessage("Hata oluştu")
    
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Çıkış', 
                                      'Uygulamadan çıkmak istediğinize emin misiniz?',
                                      QMessageBox.Yes | QMessageBox.No, 
                                      QMessageBox.No)
        if reply == QMessageBox.Yes:
            if hasattr(self, 'model'): del self.model
            if hasattr(self, 'prediction_thread') and self.prediction_thread.isRunning():
                self.prediction_thread.terminate()
                self.prediction_thread.wait()
            if hasattr(self, 'test_thread') and self.test_thread.isRunning():
                self.test_thread.terminate()
                self.test_thread.wait()
            event.accept()
        else:
            event.ignore()