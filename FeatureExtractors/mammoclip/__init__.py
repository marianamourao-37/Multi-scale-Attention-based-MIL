import os
from typing import Dict

from .efficientnet_custom import EfficientNet
from .image_encoder import HuggingfaceImageEncoder, ResNet, EfficientNet_Mammo

def load_image_encoder(config_image_encoder: Dict, multi_scale = False):
    if config_image_encoder["source"].lower() == "huggingface":
        cache_dir = config_image_encoder[
            "cache_dir"] if "cache_dir" in config_image_encoder else "~/.cache/huggingface/hub"
        gradient_checkpointing = (
            config_image_encoder[
                "gradient_checkpointing"] if "gradient_checkpointing" in config_image_encoder else False
        )
        model_type = config_image_encoder["model_type"] if "model_type" in config_image_encoder else "vit"
        _image_encoder = HuggingfaceImageEncoder(
            name=config_image_encoder["name"],
            pretrained=config_image_encoder["pretrained"],
            gradient_checkpointing=gradient_checkpointing,
            cache_dir=cache_dir,
            model_type=model_type,
            local_files_only=os.path.exists(
                os.path.join(cache_dir, f'models--{config_image_encoder["name"].replace("/", "--")}')),
        )
        
    elif (
            config_image_encoder["source"].lower() == "cnn" and (
            config_image_encoder["name"].lower() == "tf_efficientnet_b5_ns" or
            config_image_encoder["name"].lower() == "tf_efficientnetv2_s"
    )):
        _image_encoder = EfficientNet_Mammo(name=config_image_encoder["name"])
        
    elif (
            config_image_encoder["source"].lower() == "cnn" and
            config_image_encoder["name"].lower() == "tf_efficientnetv2-detect"
    ):
        _image_encoder = EfficientNet.from_pretrained("efficientnet-b2", num_classes=1, multi_scale = multi_scale)
        _image_encoder.out_dim = 1408
        
    elif (
            config_image_encoder["source"].lower() == "cnn" and
            config_image_encoder["name"].lower() == "tf_efficientnet_b5_ns-detect"
    ):
        _image_encoder = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1)
        _image_encoder.out_dim = 2048
        
    elif (
            config_image_encoder["source"].lower() == "cnn" and (
            config_image_encoder["name"].lower() == "resnet152" or
            config_image_encoder["name"].lower() == "resnet101"
    )):
        _image_encoder = ResNet(name=config_image_encoder["name"])

    else:
        raise KeyError(f"Not supported image encoder: {config_image_encoder}")
        
    return _image_encoder
    