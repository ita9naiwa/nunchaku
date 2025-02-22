import os

import folder_paths
import numpy as np
import torch
from image_gen_aux import DepthPreprocessor


class FluxDepthPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        model_paths = ["LiheYoung/depth-anything-large-hf"]
        prefix = os.path.join(folder_paths.models_dir, "checkpoints")
        local_folders = os.listdir(prefix)
        local_folders = sorted(
            [
                folder
                for folder in local_folders
                if not folder.startswith(".") and os.path.isdir(os.path.join(prefix, folder))
            ]
        )
        model_paths = local_folders + model_paths
        return {
            "required": {
                "image": ("IMAGE", {}),
                "model_path": (
                    model_paths,
                    {"tooltip": "Name of the depth preprocessor model."},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "depth_preprocess"
    CATEGORY = "SVDQuant"
    TITLE = "FLUX.1 Depth Preprocessor"

    def depth_preprocess(self, image, model_path):
        prefix = os.path.join(folder_paths.models_dir, "checkpoints")
        if os.path.exists(os.path.join(prefix, model_path)):
            model_path = os.path.join(prefix, model_path)
        processor = DepthPreprocessor.from_pretrained(model_path)
        np_image = np.asarray(image)
        np_result = np.array(processor(np_image)[0].convert("RGB"))
        out_tensor = torch.from_numpy(np_result.astype(np.float32) / 255.0).unsqueeze(0)
        return (out_tensor,)
