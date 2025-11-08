def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_device(model):
    return next(model.parameters()).device
    