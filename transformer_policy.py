import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Transformer-based feature extractor for trading data.
    Uses multi-head attention to capture temporal dependencies.

    Matches TradingEnv's observation layout (15 features total):
      [closes x10, RSI, MACD, MACD_signal, position, balance]
    """

    PRICE_SEQ_LEN = 10   # price history length
    N_INDICATORS = 3     # RSI, MACD, MACD_signal
    N_STATE = 2          # position, balance

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        expected = self.PRICE_SEQ_LEN + self.N_INDICATORS + self.N_STATE
        self.input_dim = observation_space.shape[0]
        if self.input_dim != expected:
            raise ValueError(
                f"TransformerFeaturesExtractor expects a {expected}-dim observation "
                f"(10 closes + 3 indicators + position + balance), got {self.input_dim}"
            )

        # Embedding layers
        self.price_embedding = nn.Linear(1, 64)
        self.indicator_embedding = nn.Linear(self.N_INDICATORS, 64)
        self.position_balance_embedding = nn.Linear(self.N_STATE, 64)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(self.PRICE_SEQ_LEN, 64))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(192, features_dim),  # 64*3 = 192 input features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        # Split observation: first 10 are closes, next 3 are indicators, last 2 are position/balance
        prices = observations[:, :self.PRICE_SEQ_LEN].unsqueeze(-1)            # [batch, 10, 1]
        indicators = observations[:, self.PRICE_SEQ_LEN:self.PRICE_SEQ_LEN + self.N_INDICATORS]  # [batch, 3]
        position_balance = observations[:, -self.N_STATE:]                     # [batch, 2]

        # Embed prices with positional encoding
        price_embeddings = self.price_embedding(prices)                        # [batch, 10, 64]
        price_embeddings = price_embeddings + self.positional_encoding.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply transformer to price sequence
        transformer_output = self.transformer(price_embeddings)                # [batch, 10, 64]
        pooled_prices = transformer_output.mean(dim=1)                         # [batch, 64]

        # Embed indicators
        indicator_embeddings = self.indicator_embedding(indicators)            # [batch, 64]

        # Embed position and balance
        position_balance_embeddings = self.position_balance_embedding(position_balance)  # [batch, 64]

        # Combine all features
        combined_features = torch.cat(
            [pooled_prices, indicator_embeddings, position_balance_embeddings], dim=-1
        )  # [batch, 192]

        # Final projection to features_dim (128)
        return self.output_projection(combined_features)


class TransformerTradingPolicy(ActorCriticPolicy):
    """Custom Transformer-based policy for trading."""

    def __init__(self, *args, **kwargs):
        kwargs['features_extractor_class'] = TransformerFeaturesExtractor
        kwargs['features_extractor_kwargs'] = {'features_dim': 128}

        super().__init__(*args, **kwargs)

# No need for policy registration - we use the class directly
