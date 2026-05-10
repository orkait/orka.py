import sys
import os

# Add current directory to path so orka package is findable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orka.cli import main

if __name__ == "__main__":
    # If running on Kaggle with no args, bootstrap from _KAGGLE_CONFIG
    if len(sys.argv) == 1 and os.path.exists("/kaggle/working"):
        from orka.deploy.kaggle import bootstrap_argv
        bootstrap_argv(sys.argv)
        
    sys.exit(main())
