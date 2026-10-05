from deep_multimodal.model import DeepMultimodalModel
from deep_multimodal.settings import load_settings


def test_final_configuration_gives_the_documented_model_and_eight_seeds():
    model_settings, training, config = load_settings()
    assert (model_settings.width, model_settings.layers, model_settings.dropout, model_settings.heads) == (64, 2, 0.1, 4)
    assert (training.learning_rate, training.weight_decay, training.max_epochs, training.patience, training.batch_size) == (3e-4, 1e-2, 20, 2, 4096)
    assert training.seeds == tuple(range(8)) and config["labels"] == "hard"
    assert sum(p.numel() for p in DeepMultimodalModel(model_settings).parameters()) == 296587
