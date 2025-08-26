from transformers import UperNetConfig, SwinConfig, MaskFormerSwinConfig, ConvNextV2Config, ConvNextConfig, UperNetForSemanticSegmentation

def get_model(num_labels=2, 
              channels=5, 
              image_size=384,
              backbone='swin',
              pretrained=None):

    
    if backbone == 'swin':
        backbone_config = SwinConfig(out_features=["stage1", "stage2", "stage3", "stage4"], 
                                     num_channels=channels, image_size=image_size)
    elif backbone == 'maskswin':
        backbone_config = MaskFormerSwinConfig(out_features=["stage1", "stage2", "stage3", "stage4"],
                                               num_channels=channels, image_size=image_size)
    elif backbone == 'convnext':
        backbone_config = ConvNextConfig(out_features=["stage1", "stage2", "stage3", "stage4"], 
                                         num_channels=channels, image_size=image_size)
    elif backbone == 'convnext2':
        backbone_config = ConvNextV2Config(out_features=["stage1", "stage2", "stage3", "stage4"],
                                           num_channels=channels, image_size=image_size)
    else:
        raise NotImplementedError(f"""Backbone {backbone} not recognized. Supported backbones:
                                  'swin', 'maskswin', 'convnext', 'convnext2'""")
    
    configuration = UperNetConfig(backbone_config=backbone_config, num_labels=num_labels)

    if pretrained:
        model = UperNetForSemanticSegmentation.from_pretrained(
            pretrained,
            config=configuration,
            ignore_mismatched_sizes=True)
    else:
        model = UperNetForSemanticSegmentation(config=configuration)

    model.available_backbones = """BitConfig, ConvNextConfig, ConvNextV2Config, DinatConfig,
                                FocalNetConfig, MaskFormerSwinConfig, NatConfig, ResNetConfig, 
                                SwinConfig, TimmBackboneConfig"""
    model.nametag = '___uper'

    return model