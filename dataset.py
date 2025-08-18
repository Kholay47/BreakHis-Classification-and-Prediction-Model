import os
import shutil

SOURCE_DIR = "dataset"
TARGET_DIR = "data_flattened"

os.makedirs(TARGET_DIR, exist_ok=True)

# Iterate benign and malignant
for tumor_type in ["benign", "malignant"]:
    tumor_type_path = os.path.join(SOURCE_DIR, tumor_type, "SOB")
    
    for subtype in os.listdir(tumor_type_path):
        subtype_path = os.path.join(tumor_type_path, subtype)
        if not os.path.isdir(subtype_path):
            continue
        
        target_subtype_dir = os.path.join(TARGET_DIR, subtype)
        os.makedirs(target_subtype_dir, exist_ok=True)
        
        # Iterate patient folders
        for patient_folder in os.listdir(subtype_path):
            patient_path = os.path.join(subtype_path, patient_folder)
            if not os.path.isdir(patient_path):
                continue
            
            # Iterate magnification folders (40X, 100X, etc.)
            for magnification_folder in os.listdir(patient_path):
                magnification_path = os.path.join(patient_path, magnification_folder)
                if not os.path.isdir(magnification_path):
                    continue
                
                # Copy all images
                for img_file in os.listdir(magnification_path):
                    src = os.path.join(magnification_path, img_file)
                    dst = os.path.join(target_subtype_dir, img_file)
                    
                    # Ensure unique filenames to avoid overwriting
                    if os.path.exists(dst):
                        base, ext = os.path.splitext(img_file)
                        counter = 1
                        while os.path.exists(dst):
                            dst = os.path.join(target_subtype_dir, f"{base}_{counter}{ext}")
                            counter += 1
                    
                    shutil.copy(src, dst)

print("✅ Dataset restructured successfully into", TARGET_DIR)