from .FL_losses import fluor_losses
from .HE_losses import he_losses

def get_loss_function(modality):
    if modality == 'fluorescence':
        return fluor_losses
    elif modality == 'he':
        return he_losses
    else:
        raise ValueError(f"Unknown modality: {modality}")