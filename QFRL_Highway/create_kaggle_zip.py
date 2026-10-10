import zipfile
import os

def create_zip():
    folders_to_zip = ["communication", "computing", "configs", "environment", "experiments", "federated", "quantum", "rl", "mock_torchquantum"]
    files_to_zip = ["requirements.txt"]
    
    zip_name = "kaggle_upload.zip"
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder in folders_to_zip:
            for root, _, files in os.walk(folder):
                if "__pycache__" in root:
                    continue
                for file in files:
                    if file.endswith(".pyc"):
                        continue
                    file_path = os.path.join(root, file)
                    # Force forward slashes for cross-platform compatibility (like Kaggle)
                    arcname = file_path.replace(os.sep, '/')
                    zipf.write(file_path, arcname)
                    
        for file in files_to_zip:
            if os.path.exists(file):
                zipf.write(file, file)

if __name__ == "__main__":
    create_zip()
    print("Zip created successfully with forward slashes.")
