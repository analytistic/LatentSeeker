
from src.models.LatentSeeker.modeling_LatentSeeker import LatentSeekerEncoderModel
from src.models.LatentSeeker.configuration_LatentSeeker import LatentSeekerConfig, LatentEncoderConfig
import torch
from transformers import AutoTokenizer, AutoProcessor, Qwen3VLProcessor
from src.models.LatentSeeker.processing_LatentSeeker import LatentSeekerProcessor
from src.models.LatentSeeker.modeling_LatentSeeker import LatentSeekerForConditionalGeneration

def main():
    config = LatentEncoderConfig(
        num_hidden_layers=2,
        vocab_size=100,
    )
    # model = LatentSeekerEncoderModel(config)

    processor = LatentSeekerProcessor.from_pretrained("src/models/LatentSeeker")
    config = LatentSeekerConfig()
    model = LatentSeekerForConditionalGeneration._from_config(config=config)


    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "longtext", "longtext": "The image shows a cat sitting on a windowsill."},
                {"type": "text", "text": "Describe this image."},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "The image shows a cat sitting on a windowsill."},
            ],
        },
    ]
    import json
    with open("/Users/alex/project/LatentSeeker/src/models/LatentSeeker/chat_template.json", "r") as f:
        chat_template = json.load(f)


    def get_sft_template(chat_template: dict) -> str:
        """Add {% generation %} tags around assistant content for SFT training."""
        tpl = chat_template["chat_template"]
        tpl = tpl.replace(
            "{{- '<|im_start|>' + message.role + '\\n' + content }}",
            "{{- '<|im_start|>' + message.role + '\\n' }}{% generation %}{{ content }}{% endgeneration %}",
        )
        tpl = tpl.replace(
            "{{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}",
            "{{- '<|im_start|>' + message.role + '\\n' }}{% generation %}{{ '<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}{% endgeneration %}",
        )
        return tpl


    assistant_sft_template = get_sft_template(chat_template)

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        # chat_template=assistant_sft_template,   
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
        return_assistant_tokens_mask=True,
        query_num=4,
    )
    inputs['input_ids'] = torch.where(                                                                          
        inputs['input_ids'] == 151671,                                                                          
        torch.tensor(10),                                                                                       
        torch.where(inputs['input_ids'] > 9, torch.tensor(0), inputs['input_ids'])                              
    )                                                                                                           
    inputs['longtext_input_ids'][inputs['longtext_input_ids'] > 9] = 0    


    model(**inputs)


    


if __name__ == "__main__":
    main()


