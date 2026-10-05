import torch

from deep_multimodal import constants as c
from deep_multimodal.model import DeepMultimodalModel, ModelSettings, convert_legacy_state_dict


def random_batch(size=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    positions = torch.randint(0, c.HISTORY_DAYS * c.NEIGHBOURHOOD_SIDE ** 2, (size, c.MAX_FIRE_TOKENS), generator=generator)
    positions[:, 10:] = -1
    positions[0] = -1                                                   # a row with no fire history at all
    return dict(fire_features=torch.randn(size, c.MAX_FIRE_TOKENS, 4, generator=generator), fire_positions=positions, static=torch.randn(size, len(c.STATIC_COLUMNS), generator=generator),
                row_values=torch.randn(size, len(c.ROW_COLUMNS), generator=generator), weather=torch.randn(size, c.WEATHER_DAYS, c.WEATHER_INPUTS, generator=generator),
                forecast_hours=torch.randn(size, 12, 5, generator=generator), forecast_daily=torch.randn(size, 20, generator=generator),
                forecast_missing=torch.tensor([False, True, False, False, False, False][:size]), auxiliary=torch.rand(size, 3))


def test_the_final_settings_have_the_recorded_parameter_count():
    assert sum(p.numel() for p in DeepMultimodalModel(ModelSettings()).parameters()) == 296587


def test_output_is_one_finite_logit_per_row_even_without_fire_history_or_forecast():
    model = DeepMultimodalModel(ModelSettings(width=16, layers=1, heads=2)).eval()
    logit, auxiliary = model(random_batch())
    assert logit.shape == (6,) and torch.isfinite(logit).all() and auxiliary["detections"].shape == (6, 2)


def test_changing_the_forecast_of_a_row_with_a_missing_forecast_does_not_move_its_output():
    model = DeepMultimodalModel(ModelSettings(width=16, layers=1, heads=2)).eval()
    batch = random_batch()
    changed = dict(batch, forecast_hours=batch["forecast_hours"] + torch.where(batch["forecast_missing"], 5.0, 0.0)[:, None, None])
    with torch.no_grad():
        assert torch.allclose(model(batch)[0], model(changed)[0], atol=1e-6)


def test_the_fusion_condition_starts_with_no_effect():
    model = DeepMultimodalModel(ModelSettings(width=16, layers=1, heads=2))
    for block in model.fusion_blocks:
        assert float(block.condition_projection.weight.detach().abs().sum()) == 0.0 and float(block.condition_projection.bias.detach().abs().sum()) == 0.0


def test_modality_dropout_only_acts_in_training():
    model = DeepMultimodalModel(ModelSettings(width=16, layers=1, heads=2, dropout=0.0, modality_dropout=0.9)).eval()
    batch = random_batch()
    with torch.no_grad():
        assert torch.allclose(model(batch)[0], model(batch)[0])


def test_legacy_weight_names_are_converted_to_the_names_of_this_model():
    model = DeepMultimodalModel(ModelSettings(width=16, layers=1, heads=2))
    legacy = {}
    for key, value in model.state_dict().items():
        old = key
        for new_name, old_name in (("fire_channel_projection.", "chan."), ("fusion_blocks.", "x."), ("static_scale", "s_w"), ("empty_fire_token", "empty"), ("condition_projection", "film"), ("norm_attention", "n1")):
            old = old.replace(new_name, old_name)
        legacy[old] = value
    assert "chan.weight" in legacy and "s_w" in legacy
    assert set(convert_legacy_state_dict(legacy)) == set(model.state_dict())
