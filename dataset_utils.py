import os
import io
import shutil
from PIL import Image
from typing import List, Dict, Optional, Tuple

# Extensions d'images supportées
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
DATA_ROOT = "data"

def ensure_data_root():
    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)

def is_safe_path(path: str) -> bool:
    """Vérifie que le chemin est à l'intérieur du dossier data."""
    abs_path = os.path.abspath(path)
    abs_root = os.path.abspath(DATA_ROOT)
    return abs_path.startswith(abs_root)

# --- Dataset Management ---

def list_datasets() -> List[str]:
    ensure_data_root()
    datasets = []
    with os.scandir(DATA_ROOT) as entries:
        for entry in entries:
            if entry.is_dir():
                datasets.append(entry.name)
    return sorted(datasets)

def create_dataset(name: str) -> bool:
    ensure_data_root()
    path = os.path.join(DATA_ROOT, name)
    if os.path.exists(path):
        return False
    os.makedirs(path)
    return True

def delete_dataset(name: str) -> bool:
    path = os.path.join(DATA_ROOT, name)
    if not is_safe_path(path) or not os.path.exists(path):
        return False
    shutil.rmtree(path)
    return True

def rename_dataset(old_name: str, new_name: str) -> bool:
    old_path = os.path.join(DATA_ROOT, old_name)
    new_path = os.path.join(DATA_ROOT, new_name)
    if not is_safe_path(old_path) or not is_safe_path(new_path):
        return False
    if not os.path.exists(old_path) or os.path.exists(new_path):
        return False
    os.rename(old_path, new_path)
    return True

def clone_dataset(source_name: str, new_name: str) -> bool:
    source_path = os.path.join(DATA_ROOT, source_name)
    new_path = os.path.join(DATA_ROOT, new_name)
    if not is_safe_path(source_path) or not is_safe_path(new_path):
        return False
    if not os.path.exists(source_path) or os.path.exists(new_path):
        return False
    shutil.copytree(source_path, new_path)
    return True

# --- Image Operations ---

def list_images(dataset_name: str) -> Dict:
    path = os.path.join(DATA_ROOT, dataset_name)
    if not is_safe_path(path) or not os.path.exists(path):
        return {"error": "Dataset not found"}

    images = []
    tagged_count = 0
    
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_file():
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in IMAGE_EXTENSIONS:
                        base_name = os.path.splitext(entry.name)[0]
                        txt_path = os.path.join(path, base_name + ".txt")
                        has_caption = os.path.exists(txt_path)
                        if has_caption:
                            tagged_count += 1
                        
                        images.append({
                            "name": entry.name,
                            "path": entry.path, # Absolute path for internal use
                            "rel_path": os.path.join(dataset_name, entry.name), # Relative for API
                            "has_caption": has_caption,
                            "size": entry.stat().st_size
                        })
    except Exception as e:
        return {"error": str(e)}

    images.sort(key=lambda x: x["name"])
    return {
        "dataset": dataset_name,
        "images": images,
        "total": len(images),
        "tagged": tagged_count
    }

def get_thumbnail(rel_path: str, max_size=(300, 300)) -> Optional[bytes]:
    full_path = os.path.join(DATA_ROOT, rel_path)
    if not is_safe_path(full_path) or not os.path.exists(full_path):
        return None
    try:
        with Image.open(full_path) as img:
            img.thumbnail(max_size)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=70)
            buf.seek(0)
            return buf.read()
    except Exception:
        return None

def save_image(dataset_name: str, file_data: bytes, filename: str) -> bool:
    dataset_path = os.path.join(DATA_ROOT, dataset_name)
    if not is_safe_path(dataset_path): return False
    
    file_path = os.path.join(dataset_path, filename)
    try:
        with open(file_path, "wb") as f:
            f.write(file_data)
        return True
    except:
        return False

# --- Batch Operations ---

def batch_add_trigger(dataset_name: str, image_names: List[str], trigger_word: str, position: str = "start") -> int:
    """Ajoute un trigger word au début ou à la fin des fichiers caption."""
    dataset_path = os.path.join(DATA_ROOT, dataset_name)
    if not is_safe_path(dataset_path): return 0
    
    count = 0
    for img_name in image_names:
        base_name = os.path.splitext(img_name)[0]
        txt_path = os.path.join(dataset_path, base_name + ".txt")
        
        current_content = ""
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                current_content = f.read().strip()
        
        # Avoid duplicate if already present
        if trigger_word in current_content:
            continue

        if position == "start":
            new_content = f"{trigger_word}, {current_content}" if current_content else trigger_word
        else:
            new_content = f"{current_content}, {trigger_word}" if current_content else trigger_word
            
        # Clean up double commas
        new_content = new_content.replace(", ,", ",").strip(", ")
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        count += 1
    return count

def batch_resize_images(dataset_name: str, image_names: List[str], resolution: int = 1024) -> int:
    dataset_path = os.path.join(DATA_ROOT, dataset_name)
    if not is_safe_path(dataset_path): return 0
    
    count = 0
    for img_name in image_names:
        img_path = os.path.join(dataset_path, img_name)
        try:
            with Image.open(img_path) as img:
                # Basic resize logic: fit within box, keep aspect ratio
                img.thumbnail((resolution, resolution), Image.Resampling.LANCZOS)
                
                # Save overwriting original (Backup could be added here in future)
                img.save(img_path, quality=95)
                count += 1
        except:
            pass
    return count

# --- Single File Ops ---
def read_caption(rel_path: str) -> str:
    full_path = os.path.join(DATA_ROOT, rel_path)
    base, _ = os.path.splitext(full_path)
    txt_path = base + ".txt"
    if os.path.exists(txt_path) and is_safe_path(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f: return f.read()
    return ""

def save_caption(rel_path: str, content: str) -> bool:
    full_path = os.path.join(DATA_ROOT, rel_path)
    base, _ = os.path.splitext(full_path)
    txt_path = base + ".txt"
    if is_safe_path(txt_path):
        with open(txt_path, 'w', encoding='utf-8') as f: f.write(content)
        return True
    return False