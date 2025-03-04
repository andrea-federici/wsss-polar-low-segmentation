import segmentation_models_pytorch as smp

def get_model(arch, encoder, encoder_weights, num_labels=2, channels=3, **config):
    model = smp.create_model(arch, 
                             encoder_name=encoder, 
                             in_channels=channels, 
                             classes=num_labels, 
                             encoder_weights=encoder_weights,
                             **config)
    
    model.nametag = "___smp-{arch}-{encoder}"

    return model
