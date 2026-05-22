# GVGAI JPype Fork Provenance

This vendored copy is based on:

- Repository: `https://github.com/doveliyuchen/GVGAI_jpype`
- Branch: `main`
- Source commit: `b42cfb48cf8e80ddb10456147aca8f2fbaa155b3`

Local training-data additions:

- Physics variants for `aliens`, `chopper`, and `waves` under `gym_gvgai/envs/games/*_v0/`
- Packaging dependency fixes for the JPype/Gymnasium environment
- Package-data manifest entries for VGDL game files and sprites

This directory is vendored under `training_data_gvgai/gvgai` rather than tracked as a Git submodule so the eventual merge to `main` can include a self-contained data-collection stack in one top-level folder.
