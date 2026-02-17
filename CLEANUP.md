# Cleanup: Fix Nested Clone Issue

If you have nested `ML0bag-visionsystem/ML0bag-visionsystem/` folders, run this code **once** in a Colab/Jupyter cell to fix it. It backs up your downloaded datasets, deletes the mess, clones fresh, and restores the data.

```python
import os
import shutil

base = '/home/programming'
target = os.path.join(base, 'ML0bag-visionsystem')

data_backup = '/tmp/ml0bag_data_backup'
for nested in [
os.path.join(target, 'data'),
os.path.join(target, 'ML0bag-visionsystem', 'data'),
os.path.join(target, 'ML0bag-visionsystem', 'ML0bag-visionsystem', 'data'),
]:
 if os.path.exists(os.path.join(nested, 'raw')):
  print(f"Backing up data from: {nested}")
  if os.path.exists(data_backup):
   shutil.rmtree(data_backup)
  shutil.copytree(os.path.join(nested, 'raw'), os.path.join(data_backup, 'raw'))
  break

if os.path.exists(target):
 shutil.rmtree(target)
 print(f"Deleted {target}")

os.chdir(base)
os.system('git clone https://github.com/mfibz/ML0bag-visionsystem.git')

if os.path.exists(data_backup):
 dest_data = os.path.join(target, 'data', 'raw')
 os.makedirs(os.path.join(target, 'data'), exist_ok=True)
 shutil.copytree(os.path.join(data_backup, 'raw'), dest_data)
 shutil.rmtree(data_backup)
 print(f"Restored data to {dest_data}")

os.chdir(target)
print(f"\nClean project at: {os.getcwd()}")
print(f"Contents: {os.listdir('.')}")
```

After running this:
1. Restart the runtime (Runtime > Restart runtime)
2. Close and reopen the notebook from: `/home/programming/ML0bag-visionsystem/notebooks/00_colab_training.ipynb`
3. Run all cells from the top
