import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
preprocess_path = os.path.abspath(os.path.join(current_dir, 'data'))
if preprocess_path not in sys.path:
    sys.path.append(preprocess_path)


try:
    import preprocess
    print("Preprocess module imported successfully!")
except ImportError:
    print("Failed to import preprocess. Check the path.")
