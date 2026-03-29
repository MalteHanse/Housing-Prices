from src.data_loader import load_train
from src.models import build_model
from src.preprocessing import build_preprocessor


X_train = load_train()
print(X_train)