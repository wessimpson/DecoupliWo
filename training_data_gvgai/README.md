# Training Data GVGAI

Self-contained GVGAI training-data collection stack.

Contents:

- `gvgai/`: vendored JPype GVGAI fork plus local rule variants
- `data/collect_gvgai_jpype.py`: rollout collector
- `data/export_gvgai_hf.py`: Hugging Face dataset-folder exporter
- `data/gvgai_variant_catalog.world_model.json`: full games_world_model rollout catalog
- `data/gvgai_variant_catalog.example.json`: legacy minimal example catalog
- `docs/gvgai_data_collection.md`: setup, schema, and workflow

Quick validation:

```bash
python training_data_gvgai/gvgai/build.py
pip install -r training_data_gvgai/requirements.txt
pip install -e training_data_gvgai/gvgai
python -m training_data_gvgai.data.collect_gvgai_jpype --games aliens --variants default --frames-per-variant 100
```

## Variant visual dashboard

To inspect the implemented rule variants without collecting transition shards,
generate an animated local dashboard from the repo root:

```bash
python training_data_gvgai/gvgai/build.py
pip install -r training_data_gvgai/requirements.txt
pip install -e training_data_gvgai/gvgai
python -m training_data_gvgai.data.preview_variant_dashboard \
  --steps 120 \
  --output-dir /tmp/decoupliwo_variant_live \
  --serve \
  --port 8769
```

Open `http://127.0.0.1:8769/index.html`. The page shows one animated rule-demo
rollout for each default game and implemented rule variant, with filters by
game and variant name. The default `--policy demo` mode intentionally acts out
the rule: projectile variants aim and fire repeatedly, dash variants move in
sustained directions, and speed/hazard variants keep the avatar moving so the
changed behavior is visible in the GIF.

Useful options:

```bash
# Render only a few games.
python -m training_data_gvgai.data.preview_variant_dashboard --games aliens,waves,ikaruga --serve

# Longer rollouts for inspecting later effects.
python -m training_data_gvgai.data.preview_variant_dashboard --steps 120 --serve

# Compare against the Java sampleMCTS policy instead of the scripted rule demos.
python -m training_data_gvgai.data.preview_variant_dashboard --policy mcts --steps 120 --serve

# Generate static files only, then serve with any local web server.
python -m training_data_gvgai.data.preview_variant_dashboard --output-dir /tmp/decoupliwo_variant_live
python -m http.server 8769 --bind 127.0.0.1 --directory /tmp/decoupliwo_variant_live
```

The generator writes `index.html`, GIF assets, `summary.json`, and a Java noise
log into the output directory. A non-empty `failures` list in `summary.json`
means at least one variant failed to reset or step.
