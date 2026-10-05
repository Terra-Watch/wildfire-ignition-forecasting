"""The deep multimodal network.

Four input streams are turned into tokens of one width:
1. fire history of the 7 by 7 neighbourhood over the previous 14 days (one token per occupied entry, at most 32), refined by self-attention;
2. a 60-day weather sequence read by a recurrent network (GRU); the hidden state at the end of every fifth day gives 12 tokens;
3. the day-ahead weather forecast (5 variables, 12 hourly steps) read by a 1D convolution, giving 6 tokens;
4. static-feature tokens (one per column), refined by self-attention.
A query token built from the static summary, the calendar values and three own-cell history values attends over all stream tokens through two cross-attention layers,
rescaled and shifted by the static summary and calendar values. A small gated skip path from six engineered columns is added to the output logit.
The output is a logit of the probability that a fire starts in the cell on the target day. An auxiliary head (training only) predicts the size of detections on the target day.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as functional

from deep_multimodal import constants as c
from deep_multimodal.blocks import CrossAttentionBlock, SelfAttentionBlock


@dataclass(frozen=True)
class ModelSettings:
    """Size and regularisation of the network. The final settings are in configs/final.yaml."""
    width: int = 64
    layers: int = 2
    dropout: float = 0.1
    heads: int = 4
    modality_dropout: float = 0.1        # during training, a whole weather, forecast or static stream is hidden for this share of rows
    auxiliary_weight: float = 0.2
    use_forecast: bool = True


class DeepMultimodalModel(nn.Module):
    def __init__(self, settings: ModelSettings = ModelSettings()):
        super().__init__()
        self.settings = settings
        width, heads, dropout, layers = settings.width, settings.heads, settings.dropout, settings.layers
        self.width = width
        side = c.NEIGHBOURHOOD_SIDE
        # fire history
        self.fire_channel_projection = nn.Linear(len(c.FIRE_CHANNELS), width)
        self.lag_embedding = nn.Embedding(c.HISTORY_DAYS + 1, width)
        self.position_embedding = nn.Embedding(side * side, width)
        self.empty_fire_token = nn.Parameter(torch.zeros(width))
        self.fire_blocks = nn.ModuleList(SelfAttentionBlock(width, heads, dropout) for _ in range(layers))
        # weather sequence
        self.weather_gru = nn.GRU(c.WEATHER_INPUTS, width, num_layers=min(layers, 2), batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.weather_segment_embedding = nn.Embedding(c.WEATHER_SEGMENTS, width)
        # day-ahead forecast
        self.forecast_conv = nn.Sequential(nn.Conv1d(len(c.FORECAST_VARIABLES), width, 3, padding=1), nn.GELU(), nn.Conv1d(width, width, 3, padding=1))
        self.forecast_daily_projection = nn.Linear(c.FORECAST_DAILY_VALUES, width)
        # static features
        self.static_scale = nn.Parameter(torch.randn(c.STATIC_TOKENS, width) * 0.1)
        self.static_shift = nn.Parameter(torch.zeros(c.STATIC_TOKENS, width))
        self.static_column_embedding = nn.Parameter(torch.randn(c.STATIC_TOKENS, width) * 0.02)
        self.static_blocks = nn.ModuleList(SelfAttentionBlock(width, heads, dropout) for _ in range(layers))
        # fusion
        self.stream_type_embedding = nn.Embedding(4, width)
        self.query_projection = nn.Linear(width + len(c.CALENDAR_COLUMNS) + len(c.OWN_HISTORY_COLUMNS), width)
        self.fusion_blocks = nn.ModuleList(CrossAttentionBlock(width, heads, dropout, width + len(c.CALENDAR_COLUMNS)) for _ in range(2))
        self.output_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1))
        self.skip_linear = nn.Linear(len(c.SKIP_COLUMNS), 1)
        self.skip_gate = nn.Parameter(torch.zeros(()))
        self.auxiliary_head = nn.Linear(width, 2)      # log(1 + detections) and log(1 + radiative power) on the target day

    def _streams(self, batch: dict, training: bool):
        """Tokens of the four streams with their padding masks, the query token and the conditioning vector."""
        settings, width = self.settings, self.width
        size, device = batch["fire_features"].shape[0], batch["fire_features"].device
        # fire history: the position index holds lag and neighbour position; negative means an empty slot
        padded = batch["fire_positions"] < 0
        position = batch["fire_positions"].clamp(min=0)
        side2 = c.NEIGHBOURHOOD_SIDE ** 2
        lag, cell = position // side2 + 1, position % side2
        fire = self.fire_channel_projection(batch["fire_features"]) + self.lag_embedding(torch.where(padded, torch.zeros_like(lag), lag)) + self.position_embedding(cell)
        fire = torch.cat([self.empty_fire_token.expand(size, 1, -1), fire], 1)          # an always-valid token, so no row has every key masked
        fire_padding = torch.cat([torch.zeros(size, 1, dtype=torch.bool, device=device), padded], 1)
        for block in self.fire_blocks:
            fire = block(fire, fire_padding)
        # weather: the hidden state at the end of every fifth day
        with torch.autocast(device_type="cuda", enabled=False):                          # the recurrent layers run in float32
            hidden, _ = self.weather_gru(batch["weather"].float())
        weather = hidden[:, 4::5] + self.weather_segment_embedding.weight[None]
        # forecast: six tokens from the 12 hourly steps, the daily statistics added to every token
        forecast = self.forecast_conv(batch["forecast_hours"].transpose(1, 2))
        forecast = functional.avg_pool1d(forecast, 2).transpose(1, 2) + self.forecast_daily_projection(batch["forecast_daily"])[:, None]
        # static tokens
        values = torch.cat([batch["static"], batch["row_values"][:, c.BURN_COLUMNS]], 1)
        static = values[..., None] * self.static_scale[None] + self.static_shift[None] + self.static_column_embedding[None]
        no_padding = torch.zeros(size, c.STATIC_TOKENS, dtype=torch.bool, device=device)
        for block in self.static_blocks:
            static = block(static, no_padding)
        static_summary = static.mean(1)
        # modality dropout (training only): a whole weather, forecast or static stream is hidden; the fire stream is never hidden
        def drop():
            return (torch.rand(size, device=device) < settings.modality_dropout) if training else torch.zeros(size, dtype=torch.bool, device=device)
        hide_weather, hide_forecast, hide_static = drop(), drop(), drop()
        static_summary = static_summary.masked_fill(hide_static[:, None], 0.0)
        hide_forecast = batch["forecast_missing"] | hide_forecast | (not settings.use_forecast)     # missing forecast days are masked, never zero-filled
        calendar, own_history = batch["row_values"][:, c.CALENDAR_COLUMNS], batch["row_values"][:, c.OWN_HISTORY_COLUMNS]
        condition = torch.cat([static_summary, calendar], 1)
        query = self.query_projection(torch.cat([static_summary, calendar, own_history], 1))[:, None]
        kinds = self.stream_type_embedding.weight
        parts = [(fire + kinds[0], fire_padding), (weather + kinds[1], hide_weather[:, None].expand(size, c.WEATHER_SEGMENTS)),
                 (forecast + kinds[2], hide_forecast[:, None].expand(size, c.FORECAST_TOKENS)), (static + kinds[3], hide_static[:, None].expand(size, c.STATIC_TOKENS))]
        return query, condition, parts

    def forward(self, batch: dict, training: bool = False):
        """Returns (logit of the fire probability, dictionary with the auxiliary prediction)."""
        query, condition, parts = self._streams(batch, training)
        tokens = torch.cat([tokens for tokens, _ in parts], 1)
        padding = torch.cat([mask for _, mask in parts], 1)
        for block in self.fusion_blocks:
            query = block(query, tokens, padding, condition)
        state = query[:, 0]
        logit = self.output_head(state).squeeze(1) + self.skip_gate * self.skip_linear(batch["row_values"][:, c.SKIP_COLUMNS]).squeeze(1)
        return logit, {"detections": self.auxiliary_head(state)}


def auxiliary_loss(prediction: dict, batch: dict, weight: float) -> torch.Tensor:
    """Mean squared error of log(1 + detections) and log(1 + radiative power) on the target day, times `weight`."""
    target = batch["auxiliary"]
    return weight * (functional.mse_loss(prediction["detections"][:, 0], torch.log1p(target[:, 0])) + functional.mse_loss(prediction["detections"][:, 1], torch.log1p(target[:, 1])))


# Names of the saved weights of the first release (research code) mapped to the names above.
LEGACY_NAMES = [("chan.", "fire_channel_projection."), ("lag_emb.", "lag_embedding."), ("pos_emb.", "position_embedding."), ("fire_tf.", "fire_blocks."), ("gru.", "weather_gru."),
                ("wx_seg.", "weather_segment_embedding."), ("f_conv.", "forecast_conv."), ("f_daily.", "forecast_daily_projection."), ("static_tf.", "static_blocks."),
                ("type_emb.", "stream_type_embedding."), ("q_in.", "query_projection."), ("x.", "fusion_blocks."), ("head.", "output_head."), ("skip.", "skip_linear."), ("aux_head.", "auxiliary_head."),
                ("s_w", "static_scale"), ("s_b", "static_shift"), ("s_col", "static_column_embedding"), ("empty", "empty_fire_token")]
LEGACY_BLOCK_NAMES = [(".n1.", ".norm_attention."), (".n2.", ".norm_feed_forward."), (".qkv.", ".qkv_projection."), (".out.", ".output_projection."), (".ff.", ".feed_forward."),
                      (".film.", ".condition_projection."), (".nq.", ".norm_query."), (".nk.", ".norm_tokens."), (".kv.", ".key_value_projection."), (".q.", ".query_projection.")]


def convert_legacy_state_dict(state: dict) -> dict:
    """Rename the keys of a weights file written by the research code so that it loads into DeepMultimodalModel."""
    out = {}
    for key, value in state.items():
        new = key
        for old, replacement in LEGACY_NAMES:
            if new == old or (old.endswith(".") and new.startswith(old)):
                new = replacement + new[len(old):] if old.endswith(".") else replacement
                break
        for old, replacement in LEGACY_BLOCK_NAMES:
            new = new.replace(old, replacement)
        out[new] = value
    return out
