# ui/widgets.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, 
                             QHBoxLayout, QWidget, QHeaderView, QAbstractItemView,
                             QTabWidget, QTextEdit, QSplitter, QFrame)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor
from PyQt5.QtCore import Qt
import matplotlib
matplotlib.use('Qt5Agg') # Ensure backend is set

class ImagePreviewLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 300)
        self.setStyleSheet("background-color: #f8f8f8; border: 1px solid #ddd; border-radius: 5px;")
        self.setText("Görüntü Yok")
        self.setFont(QFont('Arial', 10))
        self.setScaledContents(False)
    
    def setPixmap(self, pixmap):
        scaled_pixmap = pixmap.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        super().setPixmap(scaled_pixmap)

class ResultCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(ResultCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        self.fig.patch.set_facecolor('#f0f0f0')
        self.axes.set_title('Top-5 Tahminler', fontsize=12, fontweight='bold')
        
    def update_chart(self, classes, probabilities):
        self.axes.clear()
        y_pos = np.arange(len(classes))
        cmap = plt.cm.get_cmap('viridis')
        colors = [cmap(p) for p in probabilities]
        
        bars = self.axes.barh(y_pos, probabilities, color=colors)
        self.axes.set_yticks(y_pos)
        self.axes.set_yticklabels(classes, fontsize=10)
        self.axes.set_title('Top-5 Tahminler', fontsize=12, fontweight='bold')
        self.axes.set_xlim(0, 1.0)
        self.axes.set_xlabel('Olasılık', fontsize=10)
        self.axes.grid(True, linestyle='--', alpha=0.7)
        
        for i, bar in enumerate(bars):
            self.axes.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                          f'{probabilities[i]:.4f}', va='center', fontsize=9)
        
        self.fig.tight_layout()
        self.draw()

class TestResultsWidget(QWidget):
    """Test sonuçlarını görüntülemek için widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.test_results = []
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Tab widget oluştur
        self.tab_widget = QTabWidget()
        
        # Sonuçlar tabı
        results_tab = self.create_results_tab()
        self.tab_widget.addTab(results_tab, "Test Sonuçları")
        
        # İstatistikler tabı
        stats_tab = self.create_statistics_tab()
        self.tab_widget.addTab(stats_tab, "İstatistikler")
        
        layout.addWidget(self.tab_widget)
        
        
    def create_results_tab(self):
        """Test sonuçları tabını oluştur"""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)

        # Özet bilgileri
        self.summary_label = QLabel("Test sonucu bekleniyor...")
        self.summary_label.setStyleSheet("""
            background-color: #ecf0f1; padding: 10px; border-radius: 5px;
            font-weight: bold; color: #2c3e50;
        """)
        layout.addWidget(self.summary_label)

        # Sonuçlar tablosu
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)  # 6 değil 8 olmalı!

        # Sütun başlıklarını ayarla
        headers = ['Dosya Adı', 'Cevap', 'Tahmin', 'Güvenirlik', 'Sonuç', 'Top-5 Sınıflar', 'Top-5 Olasılıklar', 'Durum']
        self.results_table.setHorizontalHeaderLabels(headers)
    

        # Tablo ayarları
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Interactive)  # Dosya Adı
        header.setSectionResizeMode(1, QHeaderView.Interactive)  # Cevap
        header.setSectionResizeMode(2, QHeaderView.Interactive)  # Tahmin
        header.setSectionResizeMode(3, QHeaderView.Fixed)        # Güvenirlik
        header.setSectionResizeMode(4, QHeaderView.Fixed)        # Sonuç
        header.setSectionResizeMode(5, QHeaderView.Stretch)      # Top-5 Sınıflar
        header.setSectionResizeMode(6, QHeaderView.Stretch)      # Top-5 Olasılıklar
        header.setSectionResizeMode(7, QHeaderView.Fixed)        # Durum

        self.results_table.setColumnWidth(3, 100)  # Güvenirlik
        self.results_table.setColumnWidth(4, 80)   # Sonuç
        self.results_table.setColumnWidth(7, 80)   # Durum

        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSortingEnabled(True)

        layout.addWidget(self.results_table)

        return tab_widget
    def create_statistics_tab(self):
        """İstatistikler tabını oluştur"""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        
        # Splitter ile grafik ve metin alanlarını böl
        splitter = QSplitter(Qt.Vertical)
        
        # İstatistik grafikleri için canvas
        self.stats_canvas = StatsCanvas(width=8, height=6, dpi=100)
        splitter.addWidget(self.stats_canvas)
        
        # Metin tabanlı istatistikler
        stats_frame = QFrame()
        stats_frame.setFrameShape(QFrame.StyledPanel)
        stats_layout = QVBoxLayout(stats_frame)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(200)
        self.stats_text.setStyleSheet("""
            background-color: #f8f9fa; border: 1px solid #dee2e6;
            font-family: 'Consolas', 'Monaco', monospace; font-size: 10pt;
        """)
        stats_layout.addWidget(self.stats_text)
        
        splitter.addWidget(stats_frame)
        splitter.setSizes([400, 200])
        
        layout.addWidget(splitter)
        
        return tab_widget
    
    def clear_results(self):
        """Sonuçları temizle"""
        self.test_results = []
        self.results_table.setRowCount(0)
        self.summary_label.setText("Test sonucu bekleniyor...")
        self.stats_text.clear()
        self.stats_canvas.clear_plots()
    
    def update_results(self, results):
        """Test sonuçlarını güncelle"""
        self.test_results = results
        self.update_results_table(results)
        self.update_summary(results)
        self.update_statistics(results)
    
    def update_results_table(self, results):
        """Sonuçlar tablosunu güncelle"""
        self.results_table.setRowCount(len(results))
        
        for i, result in enumerate(results):
            # Dosya adı
            self.results_table.setItem(i, 0, QTableWidgetItem(result['filename']))
            
            # Cevap (Gerçek sınıf)
            true_label = result.get('true_label', 'Bilinmiyor')
            true_label_item = QTableWidgetItem(true_label if true_label else 'Bilinmiyor')
            if not true_label:
                true_label_item.setBackground(QColor('#95a5a6'))
                true_label_item.setForeground(QColor('white'))
            self.results_table.setItem(i, 1, true_label_item)
            
            # Tahmin
            prediction_item = QTableWidgetItem(result['predicted_class'])
            if result['predicted_class'] == 'HATA':
                prediction_item.setBackground(QColor('#e74c3c'))
                prediction_item.setForeground(QColor('white'))
            self.results_table.setItem(i, 2, prediction_item)
            
            # Güvenirlik
            confidence_item = QTableWidgetItem(f"{result['confidence']:.4f}")
            if result['confidence'] > 0.8:
                confidence_item.setBackground(QColor('#27ae60'))
                confidence_item.setForeground(QColor('white'))
            elif result['confidence'] > 0.6:
                confidence_item.setBackground(QColor('#f39c12'))
            elif result['confidence'] > 0.0:
                confidence_item.setBackground(QColor('#e67e22'))
                confidence_item.setForeground(QColor('white'))
            self.results_table.setItem(i, 3, confidence_item)
            
            # Sonuç (Doğru/Yanlış)
            accuracy_status = result.get('accuracy_status', 'Bilinmiyor')
            accuracy_item = QTableWidgetItem(accuracy_status)
            if accuracy_status == 'Doğru':
                accuracy_item.setBackground(QColor('#27ae60'))
                accuracy_item.setForeground(QColor('white'))
            elif accuracy_status == 'Yanlış':
                accuracy_item.setBackground(QColor('#e74c3c'))
                accuracy_item.setForeground(QColor('white'))
            else:
                accuracy_item.setBackground(QColor('#95a5a6'))
                accuracy_item.setForeground(QColor('white'))
            self.results_table.setItem(i, 4, accuracy_item)
            
            # Top-5 sınıflar
            top5_classes = ' | '.join(result.get('top5_classes', []))
            self.results_table.setItem(i, 5, QTableWidgetItem(top5_classes))
            
            # Top-5 olasılıklar
            top5_probs = ' | '.join([f"{p:.3f}" for p in result.get('top5_probs', [])])
            self.results_table.setItem(i, 6, QTableWidgetItem(top5_probs))
            
            # Durum
            if 'error' in result:
                status_item = QTableWidgetItem("Hata")
                status_item.setBackground(QColor('#e74c3c'))
                status_item.setForeground(QColor('white'))
            else:
                status_item = QTableWidgetItem("Başarılı")
                status_item.setBackground(QColor('#27ae60'))
                status_item.setForeground(QColor('white'))
            self.results_table.setItem(i, 7, status_item)
    
    def update_summary(self, results):
        """Özet bilgileri güncelle"""
        total = len(results)
        errors = sum(1 for r in results if 'error' in r)
        success = total - errors
        
        # Doğruluk istatistikleri
        correct_predictions = sum(1 for r in results if r.get('is_correct') == True)
        incorrect_predictions = sum(1 for r in results if r.get('is_correct') == False)
        unknown_ground_truth = sum(1 for r in results if r.get('is_correct') is None and 'error' not in r)
        
        if success > 0:
            avg_confidence = np.mean([r['confidence'] for r in results if 'error' not in r])
            
            summary_parts = [
                f"📊 Test Özeti: Toplam {total} dosya",
                f"✅ İşlem Başarılı: {success}",
                f"❌ İşlem Hatası: {errors}"
            ]
            
            # Doğruluk bilgilerini ekle
            if correct_predictions + incorrect_predictions > 0:
                accuracy = (correct_predictions / (correct_predictions + incorrect_predictions)) * 100
                summary_parts.extend([
                    f"🎯 Doğru Tahmin: {correct_predictions}",
                    f"❌ Yanlış Tahmin: {incorrect_predictions}",
                    f"📈 Test Accuracy: {accuracy:.2f}%"
                ])
            
            if unknown_ground_truth > 0:
                summary_parts.append(f"❓ Bilinmeyen Etiket: {unknown_ground_truth}")
            
            summary_parts.append(f"🔢 Ortalama Güven: {avg_confidence:.4f}")
            summary_text = " | ".join(summary_parts)
        else:
            summary_text = f"📊 Test Özeti: Toplam {total} dosya işlendi | ❌ Tüm dosyalarda hata oluştu"
        
        self.summary_label.setText(summary_text)
    
    def update_statistics(self, results):
        """İstatistikleri güncelle"""
        # Grafikleri güncelle
        self.stats_canvas.update_plots(results)
        
        # Metin istatistiklerini güncelle
        stats_text = self.generate_statistics_text(results)
        self.stats_text.setPlainText(stats_text)
    
    def generate_statistics_text(self, results):
        """İstatistik metnini oluştur"""
        total = len(results)
        errors = sum(1 for r in results if 'error' in r)
        success = total - errors
        
        # Doğruluk istatistikleri
        correct_predictions = sum(1 for r in results if r.get('is_correct') == True)
        incorrect_predictions = sum(1 for r in results if r.get('is_correct') == False)
        unknown_ground_truth = sum(1 for r in results if r.get('is_correct') is None and 'error' not in r)
        
        stats_lines = [
            "=" * 50,
            "TEST İSTATİSTİKLERİ",
            "=" * 50,
            f"Toplam Dosya Sayısı: {total}",
            f"Başarıyla İşlenen: {success}",
            f"İşlem Hatası: {errors}",
            f"İşlem Başarı Oranı: {(success/total)*100:.2f}%" if total > 0 else "İşlem Başarı Oranı: 0%",
            ""
        ]
        
        # Doğruluk analizini ekle
        if correct_predictions + incorrect_predictions > 0:
            accuracy = (correct_predictions / (correct_predictions + incorrect_predictions)) * 100
            stats_lines.extend([
                "DOĞRULUK ANALİZİ:",
                f"Doğru Tahmin: {correct_predictions}",
                f"Yanlış Tahmin: {incorrect_predictions}",
                f"Test Accuracy: {accuracy:.2f}%",
                f"Test Error Rate: {100-accuracy:.2f}%",
                ""
            ])
        
        if unknown_ground_truth > 0:
            stats_lines.extend([
                f"Bilinmeyen Ground Truth: {unknown_ground_truth}",
                "(Dosya adından sınıf bilgisi çıkarılamadı)",
                ""
            ])
        
        if success > 0:
            confidences = [r['confidence'] for r in results if 'error' not in r]
            stats_lines.extend([
                "GÜVEN SKORU İSTATİSTİKLERİ:",
                f"Ortalama Güven: {np.mean(confidences):.4f}",
                f"Medyan Güven: {np.median(confidences):.4f}",
                f"Minimum Güven: {np.min(confidences):.4f}",
                f"Maksimum Güven: {np.max(confidences):.4f}",
                f"Standart Sapma: {np.std(confidences):.4f}",
                ""
            ])
            
            # Tahmin edilen sınıf dağılımı
            predicted_class_counts = {}
            for r in results:
                if 'error' not in r:
                    cls = r['predicted_class']
                    predicted_class_counts[cls] = predicted_class_counts.get(cls, 0) + 1
            
            stats_lines.extend([
                "TAHMİN EDİLEN SINIF DAĞILIMI:",
                "-" * 30
            ])
            
            for cls, count in sorted(predicted_class_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / success) * 100
                stats_lines.append(f"{cls}: {count} ({percentage:.1f}%)")
            
            # Gerçek sınıf dağılımı (varsa)
            true_class_counts = {}
            for r in results:
                if r.get('true_label') and 'error' not in r:
                    cls = r['true_label']
                    true_class_counts[cls] = true_class_counts.get(cls, 0) + 1
            
            if true_class_counts:
                stats_lines.extend([
                    "",
                    "GERÇEK SINIF DAĞILIMI:",
                    "-" * 30
                ])
                
                for cls, count in sorted(true_class_counts.items(), key=lambda x: x[1], reverse=True):
                    correct_for_class = sum(1 for r in results if r.get('true_label') == cls and r.get('is_correct') == True)
                    class_accuracy = (correct_for_class / count) * 100 if count > 0 else 0
                    stats_lines.append(f"{cls}: {count} dosya (Sınıf Accuracy: {class_accuracy:.1f}%)")
        
        return "\n".join(stats_lines)
    
class StatsCanvas(FigureCanvas):
    """İstatistik grafikleri için canvas"""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(StatsCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.patch.set_facecolor('#f8f9fa')
    
    def clear_plots(self):
        """Grafikleri temizle"""
        self.fig.clear()
        self.draw()
    
    def update_plots(self, results):
        """Grafikleri güncelle"""
        self.fig.clear()
        
        success_results = [r for r in results if 'error' not in r]
        
        if not success_results:
            # Hata durumu için boş grafik
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Başarılı sonuç bulunamadı', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # 2x2 subplot düzeni
            
            # 1. Güven skoru histogramı
            ax1 = self.fig.add_subplot(2, 2, 1)
            confidences = [r['confidence'] for r in success_results]
            ax1.hist(confidences, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
            ax1.set_title('Güven Skoru Dağılımı')
            ax1.set_xlabel('Güven Skoru')
            ax1.set_ylabel('Frekans')
            ax1.grid(True, alpha=0.3)
            
            # 2. Sınıf dağılımı (pasta grafiği)
            ax2 = self.fig.add_subplot(2, 2, 2)
            class_counts = {}
            for r in success_results:
                cls = r['predicted_class']
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            # En fazla 10 sınıfı göster, gerisini "Diğer" olarak grupla
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_classes) > 10:
                top_classes = dict(sorted_classes[:9])
                others_count = sum(count for _, count in sorted_classes[9:])
                top_classes['Diğer'] = others_count
                class_counts = top_classes
            
            if class_counts:
                ax2.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%', startangle=90)
                ax2.set_title('Sınıf Dağılımı')
            
            # 3. Güven skoru trend grafiği
            ax3 = self.fig.add_subplot(2, 2, 3)
            ax3.plot(range(len(confidences)), confidences, 'o-', linewidth=1, markersize=3)
            ax3.set_title('Güven Skorları Trendi')
            ax3.set_xlabel('Dosya İndeksi')
            ax3.set_ylabel('Güven Skoru')
            ax3.grid(True, alpha=0.3)
            
            # 4. Güven aralıkları çubuk grafiği
            ax4 = self.fig.add_subplot(2, 2, 4)
            ranges = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
            range_counts = [0, 0, 0, 0, 0]
            
            for conf in confidences:
                if conf < 0.2:
                    range_counts[0] += 1
                elif conf < 0.4:
                    range_counts[1] += 1
                elif conf < 0.6:
                    range_counts[2] += 1
                elif conf < 0.8:
                    range_counts[3] += 1
                else:
                    range_counts[4] += 1
            
            colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
            ax4.bar(ranges, range_counts, color=colors, alpha=0.7, edgecolor='black')
            ax4.set_title('Güven Aralıkları Dağılımı')
            ax4.set_xlabel('Güven Aralığı')
            ax4.set_ylabel('Dosya Sayısı')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Sayıları çubukların üzerine yaz
            for i, count in enumerate(range_counts):
                if count > 0:
                    ax4.text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        self.fig.tight_layout()
        self.draw()