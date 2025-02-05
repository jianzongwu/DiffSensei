import argparse
import logging
from omegaconf import OmegaConf
import os
from datetime import datetime
import time
import gc
import itertools
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, AutoModel, LlamaTokenizer
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import transformers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler

from src.models.unet import UNetMangaModel
from src.models.resampler import Resampler
from src.models.vision_encoders.qwen_visual import QwenResampler
from src.models.mllm.peft_models import get_peft_model_with_resize_embedding
from src.models.mllm.seed_x import ContinuousLVLM
from src.models.utils import load_ckpt, load_ckpt_mllm
from src.datasets.utils import size_buckets
from src.datasets.dataset_mllm_max_ip import MangaTrainMLLMDataset, BucketBatchSampler, collate_fn
from scripts.train.scheduler import get_scheduler
from scripts.utils import print_gpu_memory_usage, get_trained_state_dict


logger = get_logger(__name__, log_level="INFO")
logging.getLogger('PIL').setLevel(logging.WARNING)


def arrange_mllm_input_image_embeds(image_embeds, target_image_embeds, config):
    arranged_embeds = []
    image_embeds = image_embeds[:, config.model.num_dummy_tokens:, :]
    target_image_embeds = target_image_embeds[:, config.model.num_dummy_tokens:, :]

    # Iterate through each sample in the batch
    for b in range(image_embeds.shape[0]):
        current_embeds = [image_embeds[b], target_image_embeds[b]]
        arranged_embeds.append(torch.stack(current_embeds, dim=0))

    # Stack all arranged embeddings for each sample in the batch along the 0th dimension
    input_image_embeds = torch.cat(arranged_embeds, dim=0)

    return input_image_embeds


def insert_mllm_output_image_embeds(image_embeds, generated_image_embeds, config):
    temp_embeds = image_embeds[:, config.model.num_dummy_tokens:, :].clone()
    
    for b in range(image_embeds.shape[0]):
        temp_embeds[b] = generated_image_embeds[b]

    image_embeds[:, config.model.num_dummy_tokens:, :] = temp_embeds

    return image_embeds


def mean_multiple_ip_embeds(image_embeds, ip_exists, config, bsz):
    """
    image_embeds: [bsz * max_num_ip_sources, num_dummy_tokens + max_num_ips * num_vision_tokens, cross_attn_dim]
    """
    ip_image_embeds = image_embeds[:, config.model.num_dummy_tokens:, :]
    ip_image_embeds = ip_image_embeds.view(bsz, config.train_data.max_num_ip_sources, config.model.max_num_ips, config.model.num_vision_tokens, -1).transpose(1, 2).contiguous() # (bsz, max_num_ips, max_num_ip_sources, num_vision_tokens, dim)
    
    ip_mask = ip_exists.unsqueeze(-1).to(ip_image_embeds.device, dtype=ip_image_embeds.dtype) # (bsz, max_num_ips, max_num_ip_sources, 1)
    if len(ip_image_embeds.shape) == 5:
        ip_mask = ip_mask.unsqueeze(-1)
    masked_ip_image_embeds = ip_image_embeds * ip_mask
    # Sum along the num_sources axis and divide by the number of valid sources (avoid dividing by zero)
    valid_sources_count = ip_mask.sum(dim=2).clamp(min=1) # shape (bsz, max_num_ips). Clamp to avoid division by zero
    ip_image_embeds = masked_ip_image_embeds.sum(dim=2) / valid_sources_count # (bsz, max_num_ips, num_vision_tokens, dim)

    ip_image_embeds = ip_image_embeds.view(bsz, config.model.max_num_ips * config.model.num_vision_tokens, -1)
    image_embeds = image_embeds.view(bsz, config.train_data.max_num_ip_sources, *image_embeds.shape[1:])[:, 0, :, :]
    image_embeds[:, config.model.num_dummy_tokens:, :] = ip_image_embeds

    return image_embeds


def main(args):
    # Load and merge config
    config = OmegaConf.load(args.config_path)
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    args_conf = OmegaConf.create(args_dict)
    config = OmegaConf.merge(config, args_conf)
    if args.resume_log_dir is not None:
        # Resume
        log_dir = args.resume_log_dir
        OmegaConf.save(config, os.path.join(log_dir, "config.yaml"))
    else:
        # Load config and create log folder
        config_name = args.config_path.split("/")[-1][:-5]
        config_folder = args.config_path.split("/")[-2]
        if config.exp_name:
            log_folder = f"{config_name}_{config.exp_name}"
        else:
            log_folder = f"{config_name}"
        
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )

    if accelerator.is_main_process and args.resume_log_dir is None:
        log_dir = os.path.join("logs", config_folder, log_folder, datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
        os.makedirs(log_dir, exist_ok=True)
        OmegaConf.save(config, os.path.join(log_dir, "config.yaml"))
    accelerator.wait_for_everyone()
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info("\n" + "\n".join([f"{k}\t{v}" for k, v in OmegaConf.to_container(config, resolve=True).items()]))
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set the training seed
    set_seed(config.seed)

    # Load pretrained models
    tokenizer = CLIPTokenizer.from_pretrained(config.model.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.model.pretrained_model_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(config.model.pretrained_model_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(config.model.pretrained_model_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(config.model.pretrained_model_path, subfolder="vae")
    noise_scheduler = DDPMScheduler.from_pretrained(config.model.pretrained_model_path, subfolder="scheduler")
    unet = UNetMangaModel.from_pretrained(config.model.pretrained_model_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(config.model.image_encoder_path)
    magi_image_encoder = AutoModel.from_pretrained(config.model.magi_image_encoder_path, trust_remote_code=True).crop_embedding_model

    # Init adapter modules
    image_proj_model = Resampler(
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=config.model.num_vision_tokens,
        num_dummy_tokens=config.model.num_dummy_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4,
        magi_embedding_dim=magi_image_encoder.config.hidden_size if magi_image_encoder is not None else None,
        use_magi=config.model.magi_image_encoder_path is not None
    )

    # Register manga condition modules in unet
    unet.set_manga_modules(
        max_num_ips=config.model.max_num_ips,
        num_vision_tokens=config.model.num_vision_tokens,
        max_num_dialogs=config.model.max_num_dialogs,
    )

    load_ckpt(image_proj_model, unet, config.model.manga_pretrained_model_path)

    # Load MLLM
    tokenizer_mllm = LlamaTokenizer.from_pretrained(config.model.seed_x_tokenizer_path)
    llm_model = get_peft_model_with_resize_embedding(
        config.model.pretrained_llm_path, 
        lora_config=config.model.lora_config,
        vocab_size=config.model.vocab_size,
        torch_dtype=config.mixed_precision,
        logger=logger
    )
    llm_model.gradient_checkpointing_enable()
    llm_model.config.use_cache = False
    logger.info("load llm model done")

    input_resampler = QwenResampler(**config.model.agent.input_resampler)
    output_resampler = QwenResampler(**config.model.agent.output_resampler)
    agent_model = ContinuousLVLM.from_pretrained(
        llm=llm_model,
        input_resampler=input_resampler,
        output_resampler=output_resampler,
    )
    logger.info('Load agent model done.')

    # Load resume checkpoints if resume
    last_ckpt_step = 0
    if args.resume_log_dir is not None:
        all_ckpt_steps = [d for d in os.listdir(log_dir) if d.startswith("step-")]
        last_ckpt_step = int(sorted(all_ckpt_steps, key=lambda x: int(x.split("-")[1]))[-1].split('-')[-1])
        checkpoint = torch.load(os.path.join(log_dir, f"step-{last_ckpt_step}", "ckpt.pth"), map_location='cpu')
        agent_model.load_state_dict(checkpoint, strict=False)

    # Optimizer
    optimizer = torch.optim.AdamW(
        agent_model.parameters(),
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.adam_beta1, config.optimizer.adam_beta2),
        weight_decay=config.optimizer.adam_weight_decay,
        eps=config.optimizer.adam_epsilon,
    )

    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=config.lr_scheduler.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )

    # DataLoader
    train_dataset = MangaTrainMLLMDataset(
        ann_path=config.train_data.ann_path,
        image_root=config.train_data.image_root,
        size_buckets=size_buckets,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        tokenizer_mllm=tokenizer_mllm,
        t_drop_rate=config.train_data.t_drop_rate,
        i_drop_rate=config.train_data.i_drop_rate,
        c_drop_rate=config.train_data.c_drop_rate,
        max_num_ips=config.model.max_num_ips,
        max_num_ip_sources=config.train_data.max_num_ip_sources,
        max_num_dialogs=config.model.max_num_dialogs,
        mask_dialog=config.train_data.mask_dialog,
        ip_self_condition_rate=config.train_data.ip_self_condition_rate,
        min_ip_height=config.train_data.min_ip_height,
        min_ip_width=config.train_data.min_ip_width,
        num_img_tokens=config.train_data.num_img_tokens,
        num_loc_tokens=config.train_data.num_loc_tokens,
        max_token_length=config.train_data.max_token_length,
        max_caption_length=config.train_data.max_caption_length,
    )
    batch_sampler = BucketBatchSampler(
        dataset=train_dataset,
        batch_size=config.train_batch_size
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=4 * accelerator.num_processes,
        collate_fn=collate_fn,
    )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device) # use fp32
    unet.to(accelerator.device) # use fp32
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    magi_image_encoder.to(accelerator.device, dtype=weight_dtype)
    image_proj_model.to(accelerator.device, dtype=weight_dtype)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    magi_image_encoder.requires_grad_(False)
    image_proj_model.requires_grad_(False)

    # Prepare everything with accelerator
    agent_model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        agent_model, optimizer, lr_scheduler, train_dataloader
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("manga")
        tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb"))

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Base batch size per device = {config.train_batch_size}")
    logger.info(f"  Batch number per epoch = {len(train_dataloader)}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")

    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=last_ckpt_step,
        desc="Step",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    global_step = 0

    agent_model.train()
    while global_step < config.max_train_steps:
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            if global_step < last_ckpt_step:
                lr_scheduler.step()
                global_step += 1
                continue
            # torch.set_printoptions(threshold=float('inf'))
            # print(f"local_process_index: {accelerator.local_process_index} input text:\n{batch['input_text']}\n")
            # print(f"local_process_index: {accelerator.local_process_index} input ids:\n{batch['input_ids']}\n")
            # print(f"local_process_index: {accelerator.local_process_index} labels:\n{batch['labels']}\n")
            # print(f"local_process_index: {accelerator.local_process_index} embeds_cmp_mask: {batch['embeds_cmp_mask']}")
            # print(f"local_process_index: {accelerator.local_process_index} embeds_gen_mask: {batch['embeds_gen_mask']}")
            with accelerator.accumulate(agent_model):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=torch.float32)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample the noise
                noise = torch.randn_like(latents)

                # Sample a random timestep for each image
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Encode IP images
                with torch.no_grad():
                    image_embeds = image_encoder(batch["ip_images"].to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2] # [bsz * max_num_ips * max_num_ip_sources, sequence_length, dim]
                    image_embeds = image_embeds.view(bsz, config.model.max_num_ips, config.train_data.max_num_ip_sources, *image_embeds.shape[1:]).transpose(1, 2).contiguous().view(bsz * config.train_data.max_num_ip_sources, config.model.max_num_ips, *image_embeds.shape[1:])
                    magi_image_embeds = magi_image_encoder(batch["magi_ip_images"].to(accelerator.device, dtype=weight_dtype)).last_hidden_state[:, 0]
                    magi_image_embeds = magi_image_embeds.view(bsz, config.model.max_num_ips, config.train_data.max_num_ip_sources, *magi_image_embeds.shape[1:]).transpose(1, 2).contiguous().view(bsz * config.train_data.max_num_ip_sources, config.model.max_num_ips, *magi_image_embeds.shape[1:])
                    image_embeds = image_proj_model(image_embeds, magi_image_embeds) # [bsz * max_num_ip_sources, num_dummy_tokens + max_num_ips * num_vision_tokens, cross_attn_dim]
                
                    target_clip_image_embeds = image_encoder(batch["target_clip_ip_images"].to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2] # [bsz * max_num_ips, sequence_length, clip_dim]
                    target_clip_image_embeds = target_clip_image_embeds.view(bsz, config.model.max_num_ips, *target_clip_image_embeds.shape[1:])
                    target_magi_image_embeds = magi_image_encoder(batch["target_magi_ip_images"].to(accelerator.device, dtype=weight_dtype)).last_hidden_state[:, 0] # [bsz * max_num_ips, magi_dim]
                    target_magi_image_embeds = target_magi_image_embeds.view(bsz, config.model.max_num_ips, *target_magi_image_embeds.shape[1:])
                    target_image_embeds = image_proj_model(target_clip_image_embeds, target_magi_image_embeds) # [bsz, num_dummy_tokens + max_num_ips * num_vision_tokens, cross_attn_dim]
                
                # Mean the max_num_ip_sources dimension
                image_embeds = mean_multiple_ip_embeds(image_embeds, batch["ip_exists"], config, bsz) # [bsz, num_dummy_tokens + max_num_ips * num_vision_tokens, cross_attn_dim]

                # MLLM
                mllm_input_image_embeds = arrange_mllm_input_image_embeds(image_embeds, target_image_embeds, config)

                mllm_output = agent_model(
                    input_ids=batch['input_ids'].to(accelerator.device),
                    attention_mask=batch['attention_mask'].to(accelerator.device),
                    labels=batch['labels'].to(accelerator.device),
                    image_embeds=mllm_input_image_embeds,
                    embeds_gen_mask=batch['embeds_gen_mask'],
                    embeds_cmp_mask=batch['embeds_cmp_mask'],
                    ids_gen_mask=batch['ids_gen_mask'].to(accelerator.device),
                    ids_cmp_mask=batch['ids_cmp_mask'].to(accelerator.device)
                )
                mllm_image_embeds = mllm_output['image_embeds']

                if mllm_output['has_image_output']:
                    image_embeds = insert_mllm_output_image_embeds(image_embeds.to(dtype=torch.float32), mllm_image_embeds, config)# + image_embeds.to(dtype=torch.float32)

                # Encode text prompt
                with torch.no_grad():
                    encoder_output = text_encoder(batch['text_input_ids'].to(accelerator.device), output_hidden_states=True)
                    encoder_output_2 = text_encoder_2(batch['text_input_ids_2'].to(accelerator.device), output_hidden_states=True)
                text_embeds = encoder_output.hidden_states[-2]
                pooled_text_embeds = encoder_output_2[0]
                text_embeds_2 = encoder_output_2.hidden_states[-2]
                text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1)

                # Concat other embeddings into text_embeds
                encoder_hidden_states = torch.cat([text_embeds, image_embeds], dim=1)

                # Prepare SDXL extra conditions
                # Transfer dialog bbox into positional embeddings and concat to add_time_ids
                add_time_ids = [
                    batch["original_size"].to(accelerator.device),
                    batch["crop_coords_top_left"].to(accelerator.device),
                    batch["target_size"].to(accelerator.device),
                ]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}
                
                # Predict the noise
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    added_cond_kwargs=unet_added_cond_kwargs,
                    cross_attention_kwargs={"bbox": batch["ip_bbox"], "aspect_ratio": latents.shape[-2] / latents.shape[-1]},
                    dialog_bbox=batch["dialog_bbox"].to(accelerator.device),
                ).sample
                
                # Compute the MSE loss
                loss_diffusion = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                loss = loss_diffusion + config.mllm_loss_weight * mllm_output['total_loss']

                # Backward
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            if config.checkpoints_total_limit != 0:
                if global_step in config.checkpointing_steps or global_step % config.checkpointing_interval == 0:
                    # Save checkpoints in checkpointing_steps and checkpointing_interval
                    # Run evaluation after saving checkpoints
                    if accelerator.is_main_process:
                        all_steps = [d for d in os.listdir(log_dir) if d.startswith("step-")]
                        all_steps = sorted(all_steps, key=lambda x: int(x.split("-")[1]))

                        if config.checkpoints_total_limit > 0 and len(all_steps) == config.checkpoints_total_limit:
                            removing_step = all_steps[0]
                            removing_step_dir = os.path.join(log_dir, removing_step)
                            ckpt_file = os.path.join(removing_step_dir, "ckpt.pth")
                            os.remove(ckpt_file)
                            logger.info(f"{len(all_steps)} checkpoint steps already exist, removing the oldest step")

                        step_dir = os.path.join(log_dir, f"step-{global_step}")
                        save_path = os.path.join(step_dir, "ckpt.pth")
                        os.makedirs(step_dir, exist_ok=True)
                        trained_state_dict = get_trained_state_dict(accelerator.unwrap_model(agent_model)) # must add, to update the state_dict of image_proj_model
                        torch.save(trained_state_dict, save_path)
                        logger.info(f"Saved state to {save_path}")
                        
            avg_loss_diffusion = accelerator.gather(loss_diffusion.unsqueeze(0)).mean().detach().item()
            avg_lm_loss = accelerator.gather(mllm_output['lm_loss'].unsqueeze(0)).mean().detach().item()
            avg_reg_loss = accelerator.gather(mllm_output['rec_loss'].unsqueeze(0)).mean().detach().item()
            
            logs = {
                "Diffusion Loss": f"{avg_loss_diffusion:.4f}",
                "LM Loss": f"{avg_lm_loss:.4f}",
                "Reg Loss": f"{avg_reg_loss:.4f}",
                "Step Time": f"{time.perf_counter() - begin:.2f}s",
            } 
            progress_bar.set_postfix(**logs)
            
            if accelerator.is_main_process:
                tb_writer.add_scalar("Diffusion Loss", avg_loss_diffusion, global_step)
                tb_writer.add_scalar("LM Loss", avg_lm_loss, global_step)
                tb_writer.add_scalar("Reg Loss", avg_reg_loss, global_step)

            if global_step >= config.max_train_steps:
                break

            begin = time.perf_counter()
            # print_gpu_memory_usage(accelerator.local_process_index)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()
    logger.info(f"The End")


if __name__ == "__main__":
    """
    nohup accelerate launch \
        --multi_gpu \
        -m scripts.train.train_mllm \
        --config_path configs/train_mllm/2024-11-7_self/self_0.0.yaml \
        --inference_config_path configs/inference/2024-10-30_negative_prompt/think_lines_pure_black_bg.yaml \
        > nohup/train_mllm_self_0.0.out 2>&1 &
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--inference_config_path", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=-1, help="-1 means no limit")
    parser.add_argument("--resume_log_dir", type=str, default=None)
    args = parser.parse_args()
    
    main(args)