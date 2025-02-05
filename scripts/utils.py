import torch


def print_gpu_memory_usage(local_rank, prefix=None):
    if prefix is not None:
        print(prefix)
    if torch.cuda.is_available():
        torch.cuda.synchronize(local_rank)

        print(f"Rank: {local_rank}, Memory Allocated: {torch.cuda.memory_allocated(local_rank) / (1024**3):.2f} GB, Max Memory Allocated: {torch.cuda.max_memory_allocated(local_rank) / (1024**3):.2f} GB")
    else:
        print("CUDA is not available. No GPU detected.")


def get_trained_state_dict(model):
    trained_state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            trained_state_dict[name] = param

    return trained_state_dict