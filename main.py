# main.py
import sys
import os

# Proje dizinini Python path'ine ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("=" * 50)
print("MULTIZOO TEST - main.py")
print("=" * 50)
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Script directory: {current_dir}")
print("=" * 50)

# rembg test
try:
    print("Testing rembg import...")
    from rembg import remove
    print("✅ SUCCESS: rembg imported successfully in main.py!")
    
    # rembg fonksiyonlarını test et
    print("Testing rembg functions...")
    print(f"✅ rembg.remove function: {remove}")
    
except ImportError as e:
    print(f"❌ FAILED: rembg import failed in main.py")
    print(f"Error: {e}")
    print("Trying to install rembg...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rembg"])
        print("✅ rembg installed successfully!")
        from rembg import remove
        print("✅ rembg import successful after installation!")
    except Exception as install_error:
        print(f"❌ Failed to install rembg: {install_error}")

print("=" * 50)

# Gerekli kütüphaneleri test et
required_packages = [
    'torch', 'torchvision', 'timm', 'PIL', 'cv2', 'numpy', 
    'matplotlib', 'PyQt5', 'pathlib', 'zipfile', 'json'
]

print("Testing required packages...")
missing_packages = []

for package in required_packages:
    try:
        if package == 'cv2':
            import cv2
        elif package == 'PIL':
            from PIL import Image
        elif package == 'PyQt5':
            from PyQt5.QtWidgets import QApplication
        elif package == 'pathlib':
            from pathlib import Path
        elif package == 'zipfile':
            import zipfile
        elif package == 'json':
            import json
        else:
            __import__(package)
        print(f"✅ {package}")
    except ImportError:
        print(f"❌ {package}")
        missing_packages.append(package)

if missing_packages:
    print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
    print("Please install missing packages before running the application.")
else:
    print("\n✅ All required packages are available!")

print("=" * 50)

# PyQt5 imports
try:
    from PyQt5.QtWidgets import QApplication
    from ui.main_window import MultiZooApp
    from ui.styles import set_app_style
    print("✅ All UI imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")
    print("Please check if all required files are in the correct directories:")
    print("  - ui/main_window.py")
    print("  - ui/widgets.py")
    print("  - ui/styles.py")
    print("  - core/model.py")
    print("  - core/prediction_thread.py")
    print("  - core/utils.py")
    sys.exit(1)

if __name__ == "__main__":
    print("Starting MultiZoo Animal Classifier...")
    
    # Uygulama dizinini kontrol et
    if not os.path.exists('ui'):
        print("❌ 'ui' directory not found!")
        print("Please make sure the application structure is correct.")
        sys.exit(1)
    
    if not os.path.exists('core'):
        print("❌ 'core' directory not found!")
        print("Please make sure the application structure is correct.")
        sys.exit(1)
    
    # Assets dizinini oluştur (yoksa)
    if not os.path.exists('assets'):
        os.makedirs('assets')
        print("📁 Created 'assets' directory")
    
    # Temp dizinini oluştur (test dosyaları için)
    if not os.path.exists('temp'):
        os.makedirs('temp')
        print("📁 Created 'temp' directory")
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    set_app_style(app)

    # Pencere ikonu ayarla (varsa)
    icon_path = os.path.join('assets', 'icon.png')
    if os.path.exists(icon_path):
        print(f"✅ Using icon: {icon_path}")
    else:
        print(f"ℹ️  Icon not found at {icon_path} (optional)")

    window = MultiZooApp()
    window.show()
    
    print("🚀 MultiZoo Animal Classifier started successfully!")
    print("Features available:")
    print("  ✅ Single image classification")
    print("  ✅ Batch test file processing")
    print("  ✅ Background removal (if rembg available)")
    print("  ✅ Image preprocessing options")
    print("  ✅ Detailed test statistics")
    print("  ✅ Export test results (JSON/CSV)")
    
    sys.exit(app.exec_())