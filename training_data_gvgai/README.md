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
