import json, re, os

fn = r'C:\Users\Asus\Desktop\Heart Disease\project_out.ipynb'
if not os.path.exists(fn):
    print('project_out.ipynb not found')
    fn = r'C:\Users\Asus\Desktop\Heart Disease\project.ipynb'
    print('Checking project.ipynb instead')

with open(fn, 'r', encoding='utf-8') as f:
    nb = json.load(f)

ANSI = re.compile(r'\x1b\[[0-9;]*[mK]')
executed = sum(1 for c in nb['cells'] if c['cell_type'] == 'code' and c.get('execution_count'))
print(f'Executed cells: {executed}')

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    for out in cell.get('outputs', []):
        if out.get('output_type') == 'error':
            ec = cell.get('execution_count')
            print(f'ERROR in cell {i} (exec_count={ec})')
            print('ename:', out.get('ename'))
            print('evalue:', out.get('evalue','')[:500])
            for line in out.get('traceback', [])[-12:]:
                print(ANSI.sub('', line))
            print()
