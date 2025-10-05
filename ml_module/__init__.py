from .lstm_model import LSTMModel
from .trainer import MLTrainer
from .scorer import MLScorer
if self.global_model_enabled:
    global_path = os.path.join(self.model_path, "global_transformer.h5")
    if os.path.exists(global_path):
        self.global_model.model = load_model(global_path)
        print("[✓] Loaded saved global Transformer model")
