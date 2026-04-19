
import os
from pathlib import Path

def test_paths():
    print(f"Current Working Directory: {os.getcwd()}")
    config_file = Path(__file__).parent.parent / "src" / "config.py"
    print(f"Config file exists: {config_file.exists()}")
    
    if config_file.exists():
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from src.config import PIPELINE_PATH, PROJECT_ROOT
        print(f"PROJECT_ROOT: {PROJECT_ROOT}")
        print(f"PIPELINE_PATH: {PIPELINE_PATH}")
        print(f"Pipeline file exists: {os.path.exists(PIPELINE_PATH)}")
    else:
        print("Could not find src/config.py")

if __name__ == "__main__":
    test_paths()
