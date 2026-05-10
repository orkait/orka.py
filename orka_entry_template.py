import base64
import os
import sys
import zipfile
import io
import shutil

# This blob will be filled by the push.sh script
ORKA_SOURCE_B64 = """{{ORKA_SOURCE_B64}}"""

def extract_orka():
    if ORKA_SOURCE_B64.startswith("{{"):
        print("Error: ORKA_SOURCE_B64 not populated. Run via push.sh")
        return False
        
    deploy_dir = "/tmp/orka_deploy"
    if os.path.exists(deploy_dir):
        shutil.rmtree(deploy_dir)
    os.makedirs(deploy_dir)
    
    zip_data = base64.b64decode(ORKA_SOURCE_B64)
    with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
        z.extractall(deploy_dir)
    
    sys.path.insert(0, deploy_dir)
    return True

if __name__ == "__main__":
    if extract_orka():
        from orka.cli import main
        # If running on Kaggle with no args, bootstrap from _KAGGLE_CONFIG
        if len(sys.argv) == 1 and os.path.exists("/kaggle/working"):
            from orka.deploy.kaggle import bootstrap_argv
            bootstrap_argv(sys.argv)
            
        sys.exit(main())
    else:
        sys.exit(1)
