import torch
import torch.nn as nn

from .descriptors.dedode_descriptor import DeDoDeDescriptor
from .detectors.dedode_detector import DeDoDeDetector
from .modules.decoder import ConvRefiner, Decoder
from .modules.encoder import VGG

MODEL_URLS = {
    "dedode_detector_L": "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_detector_L.pth",
    "dedode_descriptor_B": "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_B.pth",
}


def DeDoDeDetectorL(weights=None, num_keypoints=10000):
    NUM_PROTOTYPES = 1
    residual = True
    hidden_blocks = 8
    conv_refiner = nn.ModuleDict(
        {
            "8": ConvRefiner(
                512,
                512,
                256 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
            ),
            "4": ConvRefiner(
                256 + 256,
                256,
                128 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
            ),
            "2": ConvRefiner(
                128 + 128,
                128,
                64 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
            ),
            "1": ConvRefiner(
                64 + 64,
                64,
                1 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
            ),
        }
    )
    encoder = VGG(size="19")
    decoder = Decoder(conv_refiner)
    model = DeDoDeDetector(
        encoder=encoder, decoder=decoder, num_keypoints=num_keypoints
    )
    model.load_state_dict(
        weights
        if weights is not None
        else torch.hub.load_state_dict_from_url(MODEL_URLS["dedode_detector_L"])
    )
    return model


def DeDoDeDescriptorB(weights=None):
    NUM_PROTOTYPES = 256  # == descriptor size
    residual = True
    hidden_blocks = 5
    conv_refiner = nn.ModuleDict(
        {
            "8": ConvRefiner(
                512,
                512,
                256 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
            ),
            "4": ConvRefiner(
                256 + 256,
                256,
                128 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
            ),
            "2": ConvRefiner(
                128 + 128,
                64,
                32 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
            ),
            "1": ConvRefiner(
                64 + 32,
                32,
                1 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
            ),
        }
    )
    encoder = VGG(size="19")
    decoder = Decoder(conv_refiner, num_prototypes=NUM_PROTOTYPES)
    model = DeDoDeDescriptor(encoder=encoder, decoder=decoder)
    model.load_state_dict(
        weights
        if weights is not None
        else torch.hub.load_state_dict_from_url(MODEL_URLS["dedode_descriptor_B"])
    )
    return model
