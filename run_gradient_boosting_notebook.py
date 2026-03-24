import os, json, pathlib, traceback
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

nb_path = pathlib.Path('27_ml/gradient_boosting.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))
ctx = {'plt': plt}
errors = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    src_clean = '\n'.join(line for line in src.splitlines() if not line.strip().startswith('%') and not line.strip().startswith('!'))
    print(f'\n=== Executing cell {i} ===')
    try:
        exec(src_clean, ctx)
    except Exception as e:
        print(f'Cell {i} raised {e.__class__.__name__}: {e}')
        traceback.print_exc()
        errors.append((i, e))
        break
print('\ndone with', len(nb['cells']), 'cells. errors', len(errors))
if errors:
    for i, e in errors:
        print('failed cell', i, e)
