# Battery Cycle Life Prediction

This project trains regression models to predict lithium-ion battery cycle life using the public Stanford/MIT dataset. Both classical ML (Elastic Net, Random Forest, Extra Trees, XGBoost) and a CNN-BLSTM model on graphical time-series features are provided.

## Quickstart

1. **Environment**
   - Python 3.9+
   - Install dependencies: `pip install -r requirements.txt`
   - GPU is recommended for the CNN model (uses PyTorch + torchvision).

2. **Data**
   - Place the processed pickle files under `Data/` as configured in `config.py` (`batch1.pkl`, `batch2.pkl`, `batch3.pkl`).
   - Raw `.mat` files are included; you can regenerate the pickles with the notebooks in `data_preprocess/` if needed.

3. **Run a model**
   - Elastic Net: `python main.py --model elasticnet`
   - XGBoost: `python main.py --model xgboost`
   - Random Forest: `python main.py --model rf`
   - Extra Trees: `python main.py --model extratrees`
   - CNN-BLSTM: `python main.py --model cnn`

4. **Parameters & checkpoints**
   - Tree/XGBoost models load parameters from `config.py` (`LOAD_PARAMS`) or run a search and save best params to `params/*.json`.
   - The CNN model now saves best weights to `params/cnn_best.pth` after training and will automatically load from `CNNConfig.LOAD_PARAMS` if present.

5. **Results & artifacts**
   - Each run writes a timestamped folder under `results/` with `results.json`, metrics, and plots (prediction scatter; feature importance when available). The CNN run also logs the saved weights path in `results.json`.

## Testing

Run the lightweight regression/feature unit tests with `pytest` (see `test_*.py`).

## Notes

- Key runtime options (normalization, target log transform, split ratios) are centralized in `config.py`.
- CNN features are generated on-the-fly from raw cycles using `feature_extraction/cnn_feature.py`; normalization of targets is handled inside that extractor.
