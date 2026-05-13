from transformers import AutoConfig, AutoModel

from .configuration_LatentSeeker import LatentSeekerConfig
from .modeling_LatentSeeker import LatentSeekerForConditionalGeneration

AutoConfig.register("latent_seeker", LatentSeekerConfig)
AutoModel.register(LatentSeekerConfig, LatentSeekerForConditionalGeneration)
