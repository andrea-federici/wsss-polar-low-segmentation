from transformers import SegformerForSemanticSegmentation, SegformerConfig

def get_model(num_labels=2, channels=3, pretrained=None, **config):

    if pretrained:
        model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained,
            num_channels=channels,
            num_labels=num_labels,
            ignore_mismatched_sizes=True)
    else:
        configuration = SegformerConfig(num_labels=num_labels, num_channels=channels, **config) 
        model = SegformerForSemanticSegmentation(config=configuration)

    model.nametag = '___segformer'

    return model