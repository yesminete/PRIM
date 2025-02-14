import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
preprocess_path = os.path.abspath(os.path.join(current_dir, 'data'))
model_path = os.path.abspath(os.path.join(current_dir, 'models'))
utils_path = os.path.abspath(os.path.join(current_dir, '../utils'))
if preprocess_path not in sys.path:
    sys.path.append(preprocess_path)

if model_path not in sys.path:
    sys.path.append(model_path)

if utils_path not in sys.path:
    sys.path.append(utils_path)


try:
    import preprocess
    print("Preprocess module imported successfully!")
except ImportError:
    print("Failed to import preprocess. Check the path.")

try:
    from model_layers import *
    print("model_layers module imported successfully!")
except ImportError:
    print("Failed to import model_layers. Check the path.")

try:
    from losses import *
    print("losses module imported successfully!")
except ImportError:
    print("Failed to import losses. Check the path.")

try:
    from split import *
    print("split module imported successfully!")
except ImportError:
    print("Failed to import split. Check the path.")


