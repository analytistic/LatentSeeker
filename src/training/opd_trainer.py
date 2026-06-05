from trl.trainer.sft_trainer import SFTTrainer
from trl.experimental.gold import GOLDConfig
from transformers import PreTrainedModel, TrainerCallback
import torch
import torch.nn as nn
from transformers import DataCollator, PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin, EvalPrediction, GenerationConfig
from datasets import Dataset
from typing import Callable, Optional, Any
# from peft import PeftConfig
from trl.trainer.utils import disable_dropout_in_model


class OPSDTrainer(SFTTrainer):
    def __init__(
            self,
            model: PreTrainedModel | nn.Module | str | None = None,
            args: GOLDConfig | None = None,
            data_collator: DataCollator | None = None,
            train_dataset: Dataset | None = None,
            eval_dataset: Dataset | dict[str, Dataset] | None = None,
            processing_class: (
                PreTrainedTokenizerBase | BaseImageProcessor | ProcessorMixin | FeatureExtractionMixin | None
            ) = None,
            compute_metrics: Callable[[EvalPrediction], dict] | None = None,
            callbacks: list[TrainerCallback] | None = None,
            optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
            peft_config: Optional["PeftConfig"] = None,
            use_thinking_machines_loss: bool = False,
            fixed_teacher: bool = False,
            reason_first: bool = False,
            top_k_loss: int | None = None,
            jsd_token_clip: float | None = None,
            use_ema_teacher: bool = False,
            ema_decay: float = 0.999,
            student_thinking: bool = False,
            teacher_thinking: bool = True,
    ):
        self.model_name_or_path = model if isinstance(model, str) else model.config._name_or_path
        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

        if args.disable_dropout:
            disable_dropout_in_model(self.model)
            
        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.seq_kd = args.seq_kd
        self.use_thinking_machines_loss = use_thinking_machines_loss
        self.fixed_teacher = fixed_teacher
        self.reason_first = reason_first
        self.top_k_loss = top_k_loss
        self.jsd_token_clip = jsd_token_clip
        self.use_ema_teacher = use_ema_teacher
        self.ema_decay = ema_decay
        self._ema_params = None  # lazily initialized on first optimizer step

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
            use_cache=True,
        )
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

        # Generation config for reasoning phase (when reason_first=True)
        max_reasoning_length = getattr(args, "max_reasoning_length", 4096)
        self.reasoning_generation_config = GenerationConfig(
            max_new_tokens=max_reasoning_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
            use_cache=True,
        )
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.reasoning_generation_config.eos_token_id = self.model.generation_config.eos_token_id


    def training_step(
            self,
            model: nn.Module,
            inputs: dict[str, torch.Tensor | Any],
            num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        """
        train longtext model with on-policy self-distillation.
    


        """

        