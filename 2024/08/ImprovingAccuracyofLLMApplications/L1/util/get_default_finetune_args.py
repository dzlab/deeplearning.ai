def get_default_finetune_args():
    return {
        "learning_rate": 3e-4,
        "max_steps": 3000,
        "early_stopping": False,
        "load_best_model_at_end": False,
        "use_cached_model": False,
        "peft_args": {"r_value": 32},
    }
