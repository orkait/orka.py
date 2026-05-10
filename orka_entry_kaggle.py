import os
import sys
from pathlib import Path
import shutil

def setup_orka():
    # Kaggle mounts datasets under /kaggle/input/SLUG
    # The zip content is usually extracted or kept as zip
    ds_path = Path("/kaggle/input/orka-compiler-core")
    
    # Try to find the orka folder
    # 1. Look for unzipped folder
    src_dir = ds_path / "orka"
    if src_dir.exists():
        sys.path.insert(0, str(ds_path))
        return True
    
    # 2. Look for zip file (Kaggle sometimes keeps it zipped)
    zip_path = ds_path / "orka.zip"
    if zip_path.exists():
        import zipfile
        extract_dir = Path("/tmp/orka_extracted")
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
        sys.path.insert(0, str(extract_dir))
        return True
        
    print(f"Error: Could not find orka source in {ds_path}")
    return False

if __name__ == "__main__":
    if setup_orka():
        from orka.cli import main
        if os.path.exists("/kaggle/working"):
            from orka.deploy.kaggle import bootstrap_argv
            bootstrap_argv(sys.argv)
        sys.exit(main())
    else:
        sys.exit(1)
