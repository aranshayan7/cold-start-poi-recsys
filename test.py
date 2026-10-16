import torch
import pandas as pd
import numpy as np

print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("Pandas:", pd.__version__)
print("Device:", "mps" if torch.backends.mps.is_available() else "cpu")