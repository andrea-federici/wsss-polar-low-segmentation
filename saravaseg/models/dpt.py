from transformers import DPTForSemanticSegmentation, DPTConfig

def get_model(num_labels=2, 
              channels=3, 
              image_size=384,
              pretrained='Intel/dpt-large-ade',
              fusion_hidden_size=64,
              hidden_size=240,
              intermediate_size=1024):

    if pretrained:
        configuration = DPTConfig(num_labels=num_labels,
                                  num_channels=channels, 
                                  image_size=image_size,
                                #   fusion_hidden_size=fusion_hidden_size,
                                #   hidden_size=hidden_size,
                                #   intermediate_size=intermediate_size
                                  )
        model = DPTForSemanticSegmentation.from_pretrained(pretrained,
                                                           config=configuration,
                                                           ignore_mismatched_sizes=True)
    else:
        configuration = DPTConfig(num_labels=num_labels,
                                  num_channels=channels, 
                                  image_size=image_size,
                                  fusion_hidden_size=fusion_hidden_size,
                                  hidden_size=hidden_size,
                                  intermediate_size=intermediate_size)
        model = DPTForSemanticSegmentation(
            config=configuration)

    model.nametag = '___dpt'

    return model