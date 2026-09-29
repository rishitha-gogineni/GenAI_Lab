# Processed-data artifacts

This folder contains reproducibility metadata, not raw Yelp review text.

- `split_indices.npz` records the fixed seed-42 training and validation split.
- `vocab.json` contains the 30,000-token vocabulary built from the training subset only.
- `preprocessing_config.json` and `preprocessing_summary.json` record the cleaning and truncation choices.
