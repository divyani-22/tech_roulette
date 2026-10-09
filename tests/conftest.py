import sys
from pathlib import Path

# Add voltcap and parent directory to sys.path so 'src' and 'voltcap' can be imported anywhere
voltcap_root = Path(__file__).resolve().parent.parent
if str(voltcap_root) not in sys.path:
    sys.path.insert(0, str(voltcap_root))
if str(voltcap_root.parent) not in sys.path:
    sys.path.insert(0, str(voltcap_root.parent))
