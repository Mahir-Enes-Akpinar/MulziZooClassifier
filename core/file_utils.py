# core/file_utils.py
import os
import zipfile
import shutil
from pathlib import Path
import tempfile
from typing import List, Tuple, Optional

class TestFileHandler:
    """Test dosyalarını işlemek için yardımcı sınıf"""
    
    SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    
    def __init__(self):
        self.temp_dirs = []  # Geçici dizinleri takip et
    
    def extract_images_from_path(self, path: str) -> List[Tuple[str, str]]:
        """
        Verilen yoldan görüntü dosyalarını çıkarır.
        
        Args:
            path: Dosya veya dizin yolu
            
        Returns:
            List of (image_path, relative_name) tuples
        """
        path_obj = Path(path)
        image_files = []
        
        if path_obj.is_file():
            if path_obj.suffix.lower() == '.zip':
                # ZIP dosyası durumu
                image_files = self._extract_from_zip(path)
            elif path_obj.suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS:
                # Tek görüntü dosyası
                image_files = [(str(path_obj), path_obj.name)]
        elif path_obj.is_dir():
            # Dizin durumu
            image_files = self._extract_from_directory(path)
        else:
            raise ValueError(f"Geçersiz yol: {path}")
        
        return image_files
    
    def _extract_from_zip(self, zip_path: str) -> List[Tuple[str, str]]:
        """ZIP dosyasından görüntüleri çıkarır"""
        image_files = []
        temp_dir = tempfile.mkdtemp(prefix='multizoo_test_')
        self.temp_dirs.append(temp_dir)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # ZIP içeriğini listele
                file_list = zip_ref.namelist()
                
                for file_name in file_list:
                    # Dosya uzantısını kontrol et
                    if Path(file_name).suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS:
                        # Dosyayı geçici dizine çıkar
                        try:
                            zip_ref.extract(file_name, temp_dir)
                            full_path = os.path.join(temp_dir, file_name)
                            
                            # Dosyanın gerçekten var olduğunu kontrol et
                            if os.path.exists(full_path) and os.path.isfile(full_path):
                                # Göreli yolu temizle (dizin yapısını koru)
                                relative_name = file_name.replace('\\', '/')
                                image_files.append((full_path, relative_name))
                        except Exception as e:
                            print(f"ZIP'ten dosya çıkarma hatası {file_name}: {e}")
                            continue
        
        except zipfile.BadZipFile:
            raise ValueError(f"Geçersiz ZIP dosyası: {zip_path}")
        except Exception as e:
            raise ValueError(f"ZIP dosyası okuma hatası: {e}")
        
        if not image_files:
            raise ValueError("ZIP dosyasında desteklenen görüntü bulunamadı")
        
        return image_files
    
    def _extract_from_directory(self, dir_path: str) -> List[Tuple[str, str]]:
        """Dizinden görüntüleri toplar"""
        image_files = []
        dir_path_obj = Path(dir_path)
        
        # Recursively walk through directory
        for root, dirs, files in os.walk(dir_path):
            for file_name in files:
                if Path(file_name).suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS:
                    full_path = os.path.join(root, file_name)
                    # Göreli yolu hesapla
                    relative_path = os.path.relpath(full_path, dir_path)
                    relative_name = relative_path.replace('\\', '/')
                    image_files.append((full_path, relative_name))
        
        if not image_files:
            raise ValueError("Dizinde desteklenen görüntü bulunamadı")
        
        # Dosya adına göre sırala
        image_files.sort(key=lambda x: x[1].lower())
        
        return image_files
    
    def validate_test_file(self, path: str) -> Tuple[bool, str, int]:
        """
        Test dosyasını doğrular.
        
        Returns:
            (is_valid, message, estimated_count)
        """
        try:
            path_obj = Path(path)
            
            if not path_obj.exists():
                return False, "Dosya veya dizin bulunamadı", 0
            
            if path_obj.is_file():
                if path_obj.suffix.lower() == '.zip':
                    # ZIP dosyası kontrolü
                    try:
                        with zipfile.ZipFile(path, 'r') as zip_ref:
                            file_list = zip_ref.namelist()
                            image_count = sum(1 for f in file_list 
                                            if Path(f).suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS)
                            
                            if image_count == 0:
                                return False, "ZIP dosyasında görüntü bulunamadı", 0
                            
                            return True, f"ZIP dosyası geçerli - {image_count} görüntü", image_count
                    
                    except zipfile.BadZipFile:
                        return False, "Geçersiz ZIP dosyası", 0
                    except Exception as e:
                        return False, f"ZIP dosyası okuma hatası: {e}", 0
                
                elif path_obj.suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS:
                    return True, "Tek görüntü dosyası", 1
                else:
                    return False, "Desteklenmeyen dosya türü", 0
            
            elif path_obj.is_dir():
                # Dizin kontrolü
                image_count = 0
                for root, dirs, files in os.walk(path):
                    for file_name in files:
                        if Path(file_name).suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS:
                            image_count += 1
                
                if image_count == 0:
                    return False, "Dizinde görüntü bulunamadı", 0
                
                return True, f"Dizin geçerli - {image_count} görüntü", image_count
            
            else:
                return False, "Geçersiz dosya türü", 0
        
        except Exception as e:
            return False, f"Doğrulama hatası: {e}", 0
    
    def cleanup(self):
        """Geçici dosyaları temizler"""
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Geçici dizin temizleme hatası: {e}")
        
        self.temp_dirs.clear()
    
    def get_file_info(self, path: str) -> dict:
        """Dosya hakkında detaylı bilgi döndürür"""
        path_obj = Path(path)
        
        info = {
            'path': str(path_obj.absolute()),
            'name': path_obj.name,
            'size': 0,
            'type': 'unknown',
            'estimated_images': 0
        }
        
        try:
            if path_obj.is_file():
                info['size'] = path_obj.stat().st_size
                
                if path_obj.suffix.lower() == '.zip':
                    info['type'] = 'zip'
                    # ZIP içindeki görüntü sayısını tahmin et
                    try:
                        with zipfile.ZipFile(path, 'r') as zip_ref:
                            file_list = zip_ref.namelist()
                            info['estimated_images'] = sum(1 for f in file_list 
                                                        if Path(f).suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS)
                    except:
                        info['estimated_images'] = 0
                
                elif path_obj.suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS:
                    info['type'] = 'image'
                    info['estimated_images'] = 1
            
            elif path_obj.is_dir():
                info['type'] = 'directory'
                # Dizin boyutunu ve görüntü sayısını hesapla
                total_size = 0
                image_count = 0
                
                for root, dirs, files in os.walk(path):
                    for file_name in files:
                        file_path = os.path.join(root, file_name)
                        try:
                            total_size += os.path.getsize(file_path)
                            if Path(file_name).suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS:
                                image_count += 1
                        except:
                            continue
                
                info['size'] = total_size
                info['estimated_images'] = image_count
        
        except Exception as e:
            info['error'] = str(e)
        
        return info
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Dosya boyutunu human-readable formatta döndürür"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"