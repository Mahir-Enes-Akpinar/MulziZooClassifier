# ui/styles.py
from PyQt5.QtGui import QPalette, QColor

def set_app_style(app):
    """Uygulama renklerini ve genel stilini ayarlar."""
    primary_color = "#3498db"    # Mavi
    accent_color = "#2ecc71"     # Yeşil
    warning_color = "#e74c3c"    # Kırmızı
    bg_color = "#f5f5f5"         # Açık gri
    text_color = "#2c3e50"       # Koyu mavi-gri
    
    app.setStyleSheet(f"""
        QMainWindow, QDialog {{
            background-color: {bg_color};
            color: {text_color};
        }}
        
        QLabel {{
            color: {text_color};
        }}
        
        QPushButton {{
            border-radius: 4px;
            padding: 6px 12px;
            font-weight: bold;
            min-height: 30px;
        }}
        
        QPushButton:hover {{
            opacity: 0.8;
        }}
        
        QProgressBar {{
            border: 1px solid #bbb;
            border-radius: 3px;
            text-align: center;
            height: 20px;
        }}
        
        QProgressBar::chunk {{
            background-color: {primary_color};
            width: 1px;
        }}
        
        QFrame {{
            border-radius: 5px;
        }}
        
        QComboBox, QCheckBox {{
            padding: 5px;
        }}
        
        QSlider::groove:horizontal {{
            border: 1px solid #bbb;
            height: 8px;
            background: #f0f0f0;
            margin: 2px 0;
            border-radius: 4px;
        }}
        
        QSlider::handle:horizontal {{
            background: {primary_color};
            border: 1px solid {primary_color};
            width: 18px;
            margin: -2px 0;
            border-radius: 9px;
        }}
        
        QStatusBar {{
            background-color: #ffffff;
            color: {text_color};
            border-top: 1px solid #ddd;
        }}
    """)