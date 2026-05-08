from transformers import PreTrainedConfig
from transformers.models.qwen3_vl import Qwen3VLTextConfig, Qwen3VLConfig


class LatentEncoderConfig(Qwen3VLTextConfig):
    model_type = "longtext_encoder"
    base_config_key = "longtext_config"
    default_theta = 500000.0

    def __init__(
        self,
        query_num=32,
        compress_ratio=10,
        num_hidden_layers = 3,
        vocab_size = 10,
        deepstack_latent_indexes=[8, 16, 24],
        **kwargs,
    ):
        super().__init__(**kwargs, vocab_size=vocab_size, num_hidden_layers=num_hidden_layers)
        self.deepstack_visual_indexes = deepstack_latent_indexes
        self.query_num = query_num
        self.compress_ratio = compress_ratio
        

class LatentSeekerConfig(PreTrainedConfig):
    model_type = "latent_seeker"
    sub_configs = {
        "longtext_config": LatentEncoderConfig,
        "text_config": Qwen3VLTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        longtext_config=None,
        text_config=None,
        longtext_token_id=151671,
        tie_word_embeddings=False,
        **kwargs,
    ):
   
        if isinstance(longtext_config, dict):
            self.longtext_config = self.sub_configs["longtext_config"](**longtext_config)
        elif longtext_config is None:
            self.longtext_config = self.sub_configs["longtext_config"](num_hidden_layers=2, vocab_size=11)

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"](num_hidden_layers=2, vocab_size=11)

        self.longtext_token_id = longtext_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)

__ = ['LatentSeekerConfig']
