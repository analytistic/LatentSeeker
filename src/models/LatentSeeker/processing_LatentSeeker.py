import bisect
from typing import Any, Union, TypedDict

import numpy as np
from transformers import ProcessingKwargs
from transformers.audio_utils import load_audio
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor, Qwen3VLProcessorKwargs
from transformers.processing_utils import AllKwargsForChatTemplate, Unpack, render_jinja_template
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput
from transformers.processing_utils import ProcessorChatTemplateKwargs
import torch

LongtextInput = Union[str, list[str], list[list[str]]]







class LatentSeekerProcessorKwargs(Qwen3VLProcessorKwargs):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": True,
        },
        "videos_kwargs": {"return_metadata": True},
        "longtext_kwargs": {
            "query_num": None,
            "compress_ratio": None,
        },
    }

class LatentSeekerAllKwargsForChatTemplate(TypedDict, total=False):
    processor_kwargs: LatentSeekerProcessorKwargs
    template_kwargs: ProcessorChatTemplateKwargs


class LatentSeekerProcessor(Qwen3VLProcessor):
    def __init__(self, image_processor=None, tokenizer=None, video_processor=None,
                 chat_template=None, query_num=32, compress_ratio=None, **kwargs):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer,
                         video_processor=video_processor, chat_template=chat_template, **kwargs)

        if not isinstance(query_num, int) or query_num <= 0:
            raise ValueError("query_num must be a positive integer.")
        if compress_ratio is not None and (not isinstance(compress_ratio, (int, float)) or compress_ratio <= 0):
            raise ValueError("compress_ratio must be a positive number.")

        self.query_num = query_num
        self.compress_ratio = compress_ratio
        self.longtext_token = "<|longtext_pad|>" if not hasattr(tokenizer, "longtext_token") else tokenizer.longtext_token
        self.longtext_start_token = "<|longtext_start|>" if not hasattr(tokenizer, "longtext_start_token") else tokenizer.longtext_start_token
        self.longtext_end_token = "<|longtext_end|>" if not hasattr(tokenizer, "longtext_end_token") else tokenizer.longtext_end_token
        self.longtext_token_id = (
            tokenizer.longtext_token_id
            if getattr(tokenizer, "longtext_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.longtext_token)
        )
    

    def _merge_kwargs(self, ModelProcessorKwargs: LatentSeekerProcessorKwargs, tokenizer_init_kwargs: dict | None = None, **kwargs) -> dict[str, dict]:
        longtext_kwargs = ModelProcessorKwargs._defaults.get("longtext_kwargs", {}).copy()
        for key in longtext_kwargs:
            if key in kwargs:
                longtext_kwargs[key] = kwargs.pop(key)
        output_kwargs = super()._merge_kwargs(
            ModelProcessorKwargs,
            tokenizer_init_kwargs=tokenizer_init_kwargs,
            **kwargs
        )
        output_kwargs["longtext_kwargs"] = longtext_kwargs
        return output_kwargs

    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput = None,
        longtext: LongtextInput = None,
        **kwargs: Unpack[LatentSeekerProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            LatentSeekerProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # --- Image ---
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        # --- Video ---
        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]
            if not kwargs.get("return_metadata"):
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        # --- Longtext ---
        if longtext is not None:
            if isinstance(longtext, str):
                longtext = [[longtext]]
            elif isinstance(longtext, list) and len(longtext) > 0 and isinstance(longtext[0], str):
                longtext = [longtext]

            lt_kwargs = output_kwargs.get("longtext_kwargs", {})
            compress_ratio = lt_kwargs.get("compress_ratio") or self.compress_ratio
            query_num = lt_kwargs.get("query_num") or self.query_num

            # Tokenize each doc individually, then flat concat
            longtext_token_ids = []
            longtext_seqlens = []
            for sample in longtext:
                for doc in sample:
                    ids = self.tokenizer(doc, truncation=True, add_special_tokens=False)["input_ids"]
                    longtext_token_ids.extend(ids)
                    longtext_seqlens.append(len(ids))

            longtext_num_tokens = []
            for actual_len in longtext_seqlens:
                if compress_ratio is not None:
                    num = max(1, int(actual_len / compress_ratio))
                else:
                    num = query_num
                longtext_num_tokens.append(num)

            # cu_seqlens: document boundaries in the flat sequence
            longtext_cu_seqlens = [0]
            for seqlen in longtext_seqlens:
                longtext_cu_seqlens.append(longtext_cu_seqlens[-1] + seqlen)
        else:
            longtext_token_ids = None
            longtext_cu_seqlens = None
            longtext_num_tokens = None

        # --- Text ---
        if not isinstance(text, list):
            text = [text]
        text = text.copy()

        # Expand image placeholders
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size ** 2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        # Expand video placeholders
        if video_grid_thw is not None:
            merge_length = self.video_processor.merge_size ** 2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    metadata = video_metadata[index]
                    if metadata.fps is None:
                        metadata.fps = 24
                    curr_timestamp = self._calculate_timestamps(
                        metadata.frames_indices, metadata.fps, self.video_processor.temporal_patch_size,
                    )
                    video_placeholder = ""
                    frame_seqlen = video_grid_thw[index][1:].prod() // merge_length
                    for frame_idx in range(video_grid_thw[index][0]):
                        curr_time = curr_timestamp[frame_idx]
                        video_placeholder += f"<{curr_time:.1f} seconds>"
                        video_placeholder += (
                            self.vision_start_token + "<|placeholder|>" * frame_seqlen + self.vision_end_token
                        )
                    if f"{self.vision_start_token}{self.video_token}{self.vision_end_token}" in text[i]:
                        text[i] = text[i].replace(
                            f"{self.vision_start_token}{self.video_token}{self.vision_end_token}", video_placeholder, 1
                        )
                    else:
                        text[i] = text[i].replace(self.video_token, video_placeholder, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        # Expand longtext placeholders: each <|longtext_pad|> -> num_tokens copies
        if longtext_num_tokens is not None:
            lt_index = 0
            for i in range(len(text)):
                while self.longtext_token in text[i]:
                    num_tokens = longtext_num_tokens[lt_index]
                    text[i] = text[i].replace(
                        self.longtext_token,
                        "<|placeholder|>" * num_tokens,
                        1,
                    )
                    lt_index += 1
                text[i] = text[i].replace("<|placeholder|>", self.longtext_token)

        # Tokenize
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video", "longtext"])

        # mm_token_type_ids: 0=text, 1=image, 2=video, 3=longtext
        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            mm_token_type_ids[array_ids == self.video_token_id] = 2
            mm_token_type_ids[array_ids == self.longtext_token_id] = 3
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        longtext_outputs = {}
        if longtext_num_tokens is not None:
            longtext_outputs = {
                "longtext_input_ids": longtext_token_ids,
                "longtext_cu_seqlens": longtext_cu_seqlens,
                "longtext_num_tokens": longtext_num_tokens,
            }

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs, **longtext_outputs}, tensor_type=return_tensors)

    def _get_assistant_masks(self, input_ids: list[int]) -> list[int]:
        """Locate assistant segments via special token IDs in input_ids."""
        start_ids = self.tokenizer.convert_tokens_to_ids(["<|im_start|>", "assistant"])
        end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        mask = [0] * len(input_ids)
        in_assistant = False
        j = 0
        while j < len(input_ids):
            if not in_assistant and input_ids[j] == start_ids[0] and j + 1 < len(input_ids) and input_ids[j + 1] == start_ids[1]:
                in_assistant = True
                j += 2
            elif input_ids[j] == end_id:
                in_assistant = False
                j += 1
            else:
                if in_assistant:
                    mask[j] = 1
                j += 1
        return mask

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        chat_template: str | None = None,
        **kwargs: Unpack[ProcessorChatTemplateKwargs],
    ) -> Any:
        # --- 1. chat_template resolution ---
        if chat_template is None:
            if isinstance(self.chat_template, dict) and "default" in self.chat_template:
                chat_template = self.chat_template["default"]
            elif isinstance(self.chat_template, dict):
                raise ValueError(
                    'The processor has multiple chat templates but none of them are named "default". '
                    "You need to specify which one to use by passing the `chat_template` argument. "
                    f"Available templates are: {', '.join(self.chat_template.keys())}"
                )
            elif self.chat_template is not None:
                chat_template = self.chat_template
            else:
                raise ValueError("Cannot use apply_chat_template because this processor does not have a chat template.")
        else:
            if isinstance(self.chat_template, dict) and chat_template in self.chat_template:
                chat_template = self.chat_template[chat_template]
            pass

        # --- 2. Tokenizer fast check ---
        is_tokenizers_fast = False
        if hasattr(self, "tokenizer"):
            if hasattr(self.tokenizer, "backend"):
                is_tokenizers_fast = self.tokenizer.backend == "tokenizers"
            else:
                is_tokenizers_fast = self.tokenizer.__class__.__name__.endswith("Fast")

        # --- 3. Parameter validation ---
        if kwargs.get("continue_final_message", False):
            if kwargs.get("add_generation_prompt", False):
                raise ValueError(
                    "continue_final_message and add_generation_prompt are not compatible. "
                    "Use continue_final_message when you want the model to continue the final message, "
                    "and add_generation_prompt when you want to add a header that will prompt it "
                    "to start a new assistant message instead."
                )
            if kwargs.get("return_assistant_tokens_mask", False):
                raise ValueError("continue_final_message is not compatible with return_assistant_tokens_mask.")

        if kwargs.get("return_assistant_tokens_mask", False):
            if not is_tokenizers_fast:
                raise ValueError(
                    "`return_assistant_tokens_mask` is not possible with slow tokenizers. "
                    "Make sure you have `tokenizers` installed."
                )
            else:
                kwargs["return_offsets_mapping"] = True
            

        # --- 4. Split kwargs: template vs __call__ ---
        template_kwargs = {}
        for key in AllKwargsForChatTemplate.__annotations__["template_kwargs"].__annotations__:
            kwarg_type_defaults = AllKwargsForChatTemplate.__annotations__["template_kwargs"]
            default_value = getattr(kwarg_type_defaults, key, None)
            value = kwargs.pop(key, default_value)
            if value is not None and not isinstance(value, dict):
                template_kwargs[key] = value
        template_kwargs.update(kwargs)

        # --- 5. Sampling rate ---
        if "sampling_rate" not in template_kwargs:
            if hasattr(self, "feature_extractor") and hasattr(self.feature_extractor, "sampling_rate"):
                template_kwargs["sampling_rate"] = self.feature_extractor.sampling_rate
            else:
                template_kwargs["sampling_rate"] = 16_000

        # --- 6. Normalize batch ---
        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "content")
        ):
            is_batched = True
            conversations = conversation
        else:
            is_batched = False
            conversations = [conversation]

        # --- 7. Normalize OpenAI-style image_url ---
        for conv in conversations:
            for message in conv:
                if not isinstance(message.get("content"), list):
                    continue
                new_content = []
                for content in message["content"]:
                    if isinstance(content, dict) and content.get("type") == "image_url" and "image_url" in content:
                        image_url_info = content["image_url"]
                        url = image_url_info.get("url", "") if isinstance(image_url_info, dict) else image_url_info
                        new_content.append({"type": "image", "url": url})
                    else:
                        new_content.append(content)
                message["content"] = new_content

        # --- 8. Pop tokenize / return_dict ---
        tokenize = template_kwargs.pop("tokenize", False)
        return_dict = template_kwargs.pop("return_dict", True)

        # --- 9. Extract multimodal content from conversations ---
        if tokenize:
            batch_images, batch_videos, batch_longtext = [], [], []
            batch_audios = []
            for conv in conversations:
                images, videos, longtexts = [], [], []
                for message in conv:
                    content_list = message.get("content", [])
                    if not isinstance(content_list, list):
                        continue

                    visuals = [c for c in content_list if c.get("type") in ("image", "video")]
                    audio_fnames = [
                        c[key]
                        for c in content_list
                        for key in ["audio", "url", "path"]
                        if key in c and c.get("type") == "audio"
                    ]
                    image_fnames = [
                        v[key]
                        for v in visuals
                        for key in ["image", "url", "path", "base64"]
                        if key in v and v["type"] == "image"
                    ]
                    images.extend(image_fnames)
                    video_fnames = [
                        v[key]
                        for v in visuals
                        for key in ["video", "url", "path"]
                        if key in v and v["type"] == "video"
                    ]
                    videos.extend(video_fnames)

                    # Longtext: collect raw text
                    for c in content_list:
                        if c.get("type") == "longtext" and "longtext" in c:
                            longtexts.append(c["longtext"])

                    if not template_kwargs.get("load_audio_from_video", False):
                        for fname in audio_fnames:
                            batch_audios.append(
                                load_audio(fname, sampling_rate=template_kwargs["sampling_rate"])
                            )
                    else:
                        for fname in video_fnames:
                            batch_audios.append(
                                load_audio(fname, sampling_rate=template_kwargs["sampling_rate"])
                            )

                batch_images.append(images)
                batch_videos.append(videos)
                batch_longtext.append(longtexts)

        # --- 10. Special tokens map ---
        special_tokens_map = {}
        if hasattr(self, "tokenizer") and hasattr(self.tokenizer, "special_tokens_map"):
            special_tokens = self.tokenizer.special_tokens_map
            special_tokens_map = {k: v for k, v in special_tokens.items() if k not in template_kwargs}

        # --- 11. Render Jinja template ---
        prompt, generation_indices = render_jinja_template(
            conversations=conversations,
            chat_template=chat_template,
            **template_kwargs,
            **special_tokens_map,
        )
        if not is_batched:
            prompt = prompt[0]

        # --- 12. Tokenize and call self ---
        if tokenize:
            single_prompt = prompt[0] if is_batched else prompt
            if self.tokenizer.bos_token is not None and single_prompt.startswith(self.tokenizer.bos_token):
                kwargs["add_special_tokens"] = False

            if "do_sample_frames" not in kwargs and (
                kwargs.get("fps") is not None or kwargs.get("num_frames") is not None
            ):
                kwargs["do_sample_frames"] = True

            images_exist = any(im for im_list in batch_images for im in im_list)
            videos_exist = any(vid for vid_list in batch_videos for vid in vid_list)
            longtexts_exist = any(lt for lt_list in batch_longtext for lt in lt_list)

            out = self(
                text=prompt,
                images=batch_images if images_exist else None,
                videos=batch_videos if videos_exist else None,
                longtext=batch_longtext if longtexts_exist else None,
                audio=batch_audios if batch_audios else None,
                **kwargs,
            )

            if return_dict:
                if template_kwargs.get("return_assistant_tokens_mask", False):
                    assistant_masks = []
                    offset_mapping = out.pop("offset_mapping")
                    input_ids = out["input_ids"]
                    for i in range(len(input_ids)):
                        # current_mask = [0] * len(input_ids[i])
                        # offsets = offset_mapping[i]
                        # offset_starts = [start for start, end in offsets]
                        # for assistant_start_char, assistant_end_char in generation_indices[i]:
                        #     start_pos = bisect.bisect_left(offset_starts, assistant_start_char)
                        #     end_pos = bisect.bisect_left(offset_starts, assistant_end_char)
                        #     if not (
                        #         start_pos >= 0
                        #         and start_pos < len(offsets)
                        #         and offsets[start_pos][0] <= assistant_start_char < offsets[start_pos][1]
                        #     ):
                        #         continue
                        #     if end_pos > len(input_ids[i]):
                        #         end_pos = len(input_ids[i])
                        #     for token_id in range(start_pos, end_pos if end_pos else len(input_ids[i])):
                        #         current_mask[token_id] = 1
                        input_ids_i = input_ids[i] if isinstance(input_ids[i], list) else input_ids[i].tolist()
                        current_mask = self._get_assistant_masks(input_ids_i)
                        assistant_masks.append(current_mask)
                    out["assistant_masks"] = assistant_masks
                    out.convert_to_tensors(tensor_type=kwargs.get("return_tensors"))
                return out
            else:
                return out["input_ids"]

        return prompt

    