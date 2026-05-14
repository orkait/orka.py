import os
import sys
from pathlib import Path
import shutil

def setup_orka():
    # Kaggle mounts datasets under /kaggle/input/
    input_base = Path("/kaggle/input")
    if not input_base.exists():
        print("Error: /kaggle/input not found.")
        return False
        
    print(f"--- Kaggle Input Debug ---")
    try:
        for ds_dir in input_base.iterdir():
            print(f"  Dataset Dir: {ds_dir}")
            for item in ds_dir.iterdir():
                print(f"    Item: {item}")
    except Exception as e:
        print(f"  Debug Error: {e}")
    print(f"--- End Debug ---")

    print(f"Scanning {input_base} for Orka source...")
    for ds_dir in input_base.iterdir():
        if not ds_dir.is_dir(): continue
        
        # 1. Look for unzipped folder
        if (ds_dir / "orka").is_dir():
            print(f"Found Orka source folder in {ds_dir}")
            sys.path.insert(0, str(ds_dir))
            return True
            
        # 2. Look for zip file
        for zip_path in ds_dir.glob("*.zip"):
            print(f"Found zip archive: {zip_path}. Extracting...")
            import zipfile
            extract_dir = Path("/tmp/orka_extracted")
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_dir)
            sys.path.insert(0, str(extract_dir))
            return True
            
    print(f"Error: Could not find orka source in any directory under {input_base}")
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
