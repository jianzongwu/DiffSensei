from peft import PeftModel, get_peft_model, LoraConfig, TaskType

import torch
from transformers import LlamaForCausalLM

from src.models.mllm.modeling_llama_xformer import LlamaForCausalLM


def get_peft_model_with_resize_embedding(
    pretrained_model_name_or_path,
    lora_config=None,
    trained_parameters=None,
    trained_layers=None,
    vocab_size=None,
    torch_dtype='bf16',
    logger=None,
):
    if torch_dtype == 'bf16' or torch_dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif torch_dtype == 'fp16' or torch_dtype == 'float16':
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype)

    # print(type(peft_config.target_modules))
    if vocab_size is not None:
        model.resize_token_embeddings(vocab_size)

    if lora_config is not None:
        lora_config = LoraConfig(
            **lora_config
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.get_input_embeddings().requires_grad_(True)
        peft_model.get_output_embeddings().requires_grad_(True)

        trainable_params, all_param = peft_model.get_nb_trainable_parameters()
    elif trained_layers is not None:
        peft_model = model
        
        all_param = 0
        trainable_params = 0
        if trained_layers == "later_10":
            for name, param in peft_model.named_parameters():
                is_train = False
                parts = name.split('.')
                try:
                    layer_number = int(parts[2])
                    if layer_number >= 3 * model.config.num_hidden_layers // 4:
                        is_train = True
                    else:
                        is_train = False
                except Exception as e:
                    is_train = True
                
                if is_train == True:
                    trainable_params += param.numel() * param.element_size()
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
                all_param += param.numel() * param.element_size()
    elif trained_parameters is not None:
        peft_model = model
        all_param = 0
        trainable_params = 0
        for name, param in peft_model.named_parameters():
            is_train = False
            for trained_post_fix in trained_parameters:
                if trained_post_fix in name:
                    param.requires_grad_(True)
                    trainable_params += param.numel() * param.element_size()
                    is_train = True
                    break
            if is_train == False:
                param.requires_grad_(False)
            all_param += param.numel() * param.element_size()
    else:
        peft_model = model

        all_param = 0
        trainable_params = 0
        for param in model.parameters():
            if param.requires_grad is True:
                trainable_params += param.numel() * param.element_size()
            all_param += param.numel() * param.element_size()


    if logger is not None:
        logger.info(
            f"trainable params: {trainable_params / (1024 * 1024):.2f} MB || "
            f"all params: {all_param / (1024 * 1024):.2f} MB || "
            f"trainable%: {100 * trainable_params / all_param:.4f}"
        )

    return peft_model
