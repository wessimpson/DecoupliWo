# World Model

This folder contains the training, inference, dataset, and model code for the diffusion-style world model.

## Folder Structure

`world_model/train.py`

- Main training entrypoint.
- Responsibilities:
  - parse training arguments
  - build dataset and dataloader
  - build `WorldModel`
  - run optimization, logging, and checkpoint saving

`world_model/inference.py`

- Inference / rollout generation entrypoint.
- Responsibilities:
  - load one sample context from the dataset
  - build `WorldModel`
  - run iterative denoising to predict future frames
  - decode outputs and save them as images / gif

`world_model/dataset.py`

- Dataset and preprocessing utilities for world-model training/inference.
- Responsibilities:
  - scan rollout shards and build a lightweight global index
  - lazily open shard files per worker
  - load only the requested sample window in `__getitem__`
  - apply sample transforms such as `preprocess(...)`

`world_model/causal_latent_dit.py`

- Separate experimental / alternative causal latent transformer-style model.
- Not part of the main diffusion training path in `train.py`.

## Model Subfolder

`world_model/model/world_model.py`

- High-level model wrapper.
- Responsibilities:
  - compose the VAE and diffuser into one `WorldModel`
  - expose training-facing methods such as:
    - `encode_context_chunked(...)`
    - `encode_target_chunked(...)`
    - `encode_actions(...)`
    - `predict_noise(...)`
    - checkpoint helpers

`world_model/model/__init__.py`

- Small package export file for model-level imports.

## Net Subfolder

`world_model/model/net/diffuser.py`

- Diffusion backbone wrapper.
- Responsibilities:
  - build the conditional UNet
  - build the action embedding
  - build and hold the noise scheduler

`world_model/model/net/vae.py`

- WanVAE implementation and checkpoint loading.
- Responsibilities:
  - encode video/frame tensors into latent space
  - decode latents back into pixel space

`world_model/model/net/__init__.py`

- Small package export file for net-level imports.

## Data Flow

Training data flow:

1. `dataset.py` loads one rollout window.
2. The dataset transform converts raw frames/actions into:
   - `context_frames`
   - `target_frame`
   - `last_action`
3. `train.py` sends that batch into `WorldModel`.
4. `WorldModel` encodes frames into VAE latents.
5. `Diffuser` predicts noise on the target latent.
6. `train.py` computes loss, backpropagates, logs, and saves checkpoints.

Inference data flow:

1. `dataset.py` provides a context window.
2. `WorldModel` encodes the context.
3. `Diffuser` denoises a sampled target latent over several steps.
4. `VAE` decodes the predicted latent back into an image.

## Ownership Guide

Use `train.py` for:

- experiment arguments
- optimizer / scheduler setup
- training loop orchestration
- checkpoint cadence

Use `dataset.py` for:

- shard indexing
- sample loading
- sample-level preprocessing / transforms

Use `world_model.py` for:

- model-side latent preparation logic
- VAE + diffuser coordination
- model-facing utilities that training and inference both need

Use `diffuser.py` and `vae.py` for:

- low-level network implementation details
