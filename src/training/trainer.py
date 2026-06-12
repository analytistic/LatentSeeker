"""LatentSeeker basic trainer classes.

For builder logic see train.py.
"""

import transformers

class Trainer(transformers.Trainer):
    """Basic Trainer that inherits from transformers.Trainer.

    This is the default trainer used if no custom trainer is specified.
    """