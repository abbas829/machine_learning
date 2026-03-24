import json
from pathlib import Path
path = Path('27_ml/gradient_boosting.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))
updated = False
for c in nb['cells']:
    if c.get('cell_type') == 'code' and ''.join(c.get('source','')).strip().startswith('# Fit without early stopping'):
        c['source'] = [
            'import xgboost as xgb\n',
            '\n',
            '# XGBoost model definition\n',
            'xgb_model = xgb.XGBRegressor(\n',
            '    n_estimators=500,\n',
            '    learning_rate=0.1,\n',
            '    max_depth=6,\n',
            '    subsample=0.8,\n',
            '    colsample_bytree=0.8,\n',
            '    random_state=42,\n',
            '    n_jobs=-1,\n',
            '    verbosity=0\n',
            ')\n',
            '\n',
            '# Fit without early stopping (compatible with this XGBoost version)\n',
            'xgb_model.fit(\n',
            '    X_tr, y_tr,\n',
            '    eval_set=[(X_tr, y_tr), (X_val, y_val)],  # Track train and validation loss\n',
            '    verbose=False\n',
            ')\n',
            '\n',
            'print(f"Model fitted with {xgb_model.n_estimators} estimators")\n',
            'print(f"Training completed successfully")\n'
        ]
        updated = True
        break
if not updated:
    raise SystemExit('No matching cell to patch')
path.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print('updated xgb_model cell')

