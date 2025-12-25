import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import jax.image
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


@dataclasses.dataclass(frozen=True)
class Gemma3WeightLoader(WeightLoader):
    """Loads weights from Google's Gemma3 4B VLM checkpoint.

    Gemma3 is a VLM with SigLIP vision encoder + Gemma3 language model.
    Downloaded from Kaggle: google/gemma-3/flax/gemma3-4b-it

    Key differences from PaliGemma:
    1. **Tokenizer**: Gemma3 uses 262k vocab (improved CJK), PaliGemma uses 257k
       - Requires updating the entire pipeline to use Gemma3 tokenizer
       - Both VLM and action expert use the same 262k vocab

    2. **Checkpoint structure**: Different naming convention
       - Gemma3: SigLiPFromPatches_0/... → OpenPI: PaliGemma/img/...
       - Gemma3: transformer/... → OpenPI: PaliGemma/llm/...

    3. **Layer format**: Gemma3 doesn't use nn.scan (each layer separate),
       while OpenPI uses nn.scan (layers stacked in first dimension)

    4. **Image size**: Gemma3 uses 1024x1024 images, OpenPI uses 256x256
       - Position embeddings are interpolated from 64x64 → 16x16 grid

    Only loads VLM (expert 0) weights. Action expert (expert 1) is initialized
    randomly and trained from scratch via flow matching (Pi0/Pi0.5 paper).
    """

    target_img_size: int = 224  # Matches OpenPI data pipeline resize

    def load(self, params: at.Params) -> at.Params:
        import orbax.checkpoint as ocp
        from pathlib import Path

        # Load from Kaggle cache
        ckpt_path = Path("/root/.cache/kagglehub/models/google/gemma-3/flax/gemma3-4b-it/1/gemma3-4b-it")
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Gemma3 checkpoint not found at {ckpt_path}. "
                f"Download using: kagglehub.model_download('google/gemma-3/flax/gemma3-4b-it')"
            )

        logger.info(f"Loading Gemma3 checkpoint from {ckpt_path}")
        checkpointer = ocp.StandardCheckpointer()
        gemma3_checkpoint = checkpointer.restore(ckpt_path)

        # Transform Gemma3 structure to OpenPI PaliGemma structure
        # Interpolates position embeddings from 1024x1024 to target_img_size
        loaded_params = _transform_gemma3_to_openpi(gemma3_checkpoint, target_img_size=self.target_img_size)

        # Merge: loads VLM weights, fills action expert with random init
        return _merge_params(loaded_params, params, missing_regex=".*")


def _transform_gemma3_to_openpi(gemma3_checkpoint: at.Params, target_img_size: int = 224) -> at.Params:
    """Transform Gemma3 checkpoint structure to OpenPI PaliGemma structure.

    Gemma3 checkpoint structure:
        SigLiPFromPatches_0/siglip_encoder/...
        SigLiPFromPatches_0/siglip_encoder/Transformer/encoderblock_N/...  (N=0..26, separate)
        transformer/embedder/...
        transformer/layer_N/...  (N=0..27, separate)
        transformer/final_norm/...

    OpenPI PaliGemma structure (with nn.scan):
        PaliGemma/img/...
        PaliGemma/img/Transformer/encoderblock/...  (first dim = 27 layers)
        PaliGemma/llm/embedder/...
        PaliGemma/llm/layers/...  (first dim = depth layers)
        PaliGemma/llm/final_norm/...

    Args:
        gemma3_checkpoint: Raw Gemma3 checkpoint from Orbax
        target_img_size: Target image size (default 224 for OpenPI, Gemma3 uses 1024)

    Returns:
        Transformed parameters matching OpenPI structure
    """
    flat_gemma3 = flax.traverse_util.flatten_dict(gemma3_checkpoint, sep="/")

    # Separate vision and language parameters
    vision_params = {}
    lang_params = {}

    for key, value in flat_gemma3.items():
        if key.startswith("SigLiPFromPatches_0/siglip_encoder/"):
            # Transform vision encoder keys
            # Remove 'SigLiPFromPatches_0/siglip_encoder/' prefix
            new_key = key.replace("SigLiPFromPatches_0/siglip_encoder/", "")
            vision_params[new_key] = value
        elif key.startswith("transformer/"):
            # Transform language model keys
            # Remove 'transformer/' prefix
            new_key = key.replace("transformer/", "")
            lang_params[new_key] = value

    # Stack vision encoder layers (encoderblock_N → encoderblock with first dim)
    # Also interpolates position embeddings to target image size
    vision_transformed = _stack_vision_layers(vision_params, target_img_size=target_img_size)

    # Stack language model layers (layer_N → layers with first dim)
    lang_transformed = _stack_language_layers(lang_params)

    # Wrap in OpenPI structure
    openpi_params = {
        "PaliGemma": {
            "img": vision_transformed,
            "llm": lang_transformed,
        }
    }

    return openpi_params


def _interpolate_pos_embeddings(pos_emb: np.ndarray, target_num_patches: int) -> np.ndarray:
    """Interpolate position embeddings to match target image size.

    Gemma3 uses 1024x1024 images (4096 patches), but we may want to use smaller images.
    This function resizes the position embeddings via 2D interpolation.

    Args:
        pos_emb: Position embeddings of shape [1, num_patches, emb_dim]
        target_num_patches: Target number of patches (e.g., 256 for 224x224 images)

    Returns:
        Interpolated position embeddings of shape [1, target_num_patches, emb_dim]
    """
    if pos_emb.shape[1] == target_num_patches:
        return pos_emb  # Already the right size

    # Extract dimensions
    batch, num_patches, emb_dim = pos_emb.shape
    assert batch == 1, "Expected batch size 1 for position embeddings"

    # Calculate grid sizes
    src_grid_size = int(np.sqrt(num_patches))
    tgt_grid_size = int(np.sqrt(target_num_patches))

    assert src_grid_size ** 2 == num_patches, f"num_patches {num_patches} is not a perfect square"
    assert tgt_grid_size ** 2 == target_num_patches, f"target_num_patches {target_num_patches} is not a perfect square"

    logger.info(f"Interpolating position embeddings from {src_grid_size}x{src_grid_size} to {tgt_grid_size}x{tgt_grid_size}")

    # Reshape to 2D grid: [1, h, w, emb_dim]
    pos_emb_2d = pos_emb.reshape(1, src_grid_size, src_grid_size, emb_dim)

    # Interpolate using JAX (bilinear)
    pos_emb_resized = jax.image.resize(
        pos_emb_2d,
        shape=(1, tgt_grid_size, tgt_grid_size, emb_dim),
        method="bilinear"
    )

    # Reshape back to [1, target_num_patches, emb_dim]
    pos_emb_out = pos_emb_resized.reshape(1, target_num_patches, emb_dim)

    return np.array(pos_emb_out)


def _stack_vision_layers(vision_params: dict, target_img_size: int = 224) -> dict:
    """Stack SigLIP encoder layers from separate encoderblock_N to scanned format.

    Args:
        vision_params: Dictionary of vision encoder parameters
        target_img_size: Target image size (default 224 for OpenPI)
    """
    result = {}
    layer_params = {}

    # Infer patch size from convolutional embedding kernel.
    embedding_kernel = vision_params.get("embedding/kernel")
    if embedding_kernel is None:
        raise ValueError("Missing embedding/kernel in SigLIP parameters; cannot infer patch size.")
    patch_height, patch_width = map(int, embedding_kernel.shape[:2])
    if patch_height != patch_width:
        raise ValueError("Non-square patches are not supported.")
    if target_img_size % patch_height != 0:
        raise ValueError(
            f"Target image size {target_img_size} is not divisible by patch size {patch_height}."
        )
    target_grid = target_img_size // patch_height
    target_num_patches = target_grid * target_grid

    for key, value in vision_params.items():
        if key.startswith("Transformer/encoderblock_"):
            # Extract layer number and param path
            # e.g., "Transformer/encoderblock_5/LayerNorm_0/scale" → layer=5, path="LayerNorm_0/scale"
            parts = key.split("/")
            layer_num = int(parts[1].split("_")[1])
            param_path = "/".join(parts[2:])

            if param_path not in layer_params:
                layer_params[param_path] = {}
            layer_params[param_path][layer_num] = value
        elif key == "pos_embedding":
            # Interpolate position embeddings to target image size
            result[key] = _interpolate_pos_embeddings(value, target_num_patches)
        else:
            # Other non-layer params (embedding, encoder_norm, etc.)
            result[key] = value

    # Stack layer params
    for param_path, layers_dict in layer_params.items():
        # Sort by layer number and stack
        sorted_layers = [layers_dict[i] for i in sorted(layers_dict.keys())]
        result[f"Transformer/encoderblock/{param_path}"] = np.stack(sorted_layers, axis=0)

    return result


def _stack_language_layers(lang_params: dict) -> dict:
    """Stack Gemma3 transformer layers from separate layer_N to scanned format.

    Gemma3 uses 262k vocab, which is directly compatible with Gemma3-based models.
    No vocab truncation is needed - the model's vocab_size config will match.
    """
    result = {}
    layer_params = {}

    for key, value in lang_params.items():
        if key.startswith("layer_"):
            # Extract layer number and param path
            # e.g., "layer_5/attn/q_einsum/w" → layer=5, path="attn/q_einsum/w"
            parts = key.split("/")
            layer_num = int(parts[0].split("_")[1])
            param_path = "/".join(parts[1:])

            if param_path not in layer_params:
                layer_params[param_path] = {}
            layer_params[param_path][layer_num] = value
        elif key == "embedder/input_embedding":
            # Load embeddings as-is (262k vocab for Gemma3)
            # The model's Config.vocab_size must match this
            gemma3_vocab_size = value.shape[0]
            logger.info(f"Loading {gemma3_vocab_size} token embeddings from Gemma3 checkpoint")
            result[key] = value
        else:
            # Other non-layer params (final_norm, etc.)
            result[key] = value

    # Stack layer params
    for param_path, layers_dict in layer_params.items():
        # Sort by layer number and stack
        sorted_layers = [layers_dict[i] for i in sorted(layers_dict.keys())]
        result[f"layers/{param_path}"] = np.stack(sorted_layers, axis=0)

    return result

