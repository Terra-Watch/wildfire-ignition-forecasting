"""Execute a notebook cell by cell and save it after every cell, so its outputs appear in the file as they are produced.
No time limit per cell. Usage: python scripts/run_notebook.py notebooks/02_labels.ipynb (needs the .env variables set)."""
import sys

import nbformat
from nbclient import NotebookClient

path = sys.argv[1]
nb = nbformat.read(path, as_version=4)


async def save(cell, cell_index, execute_reply=None, **kw):
    nbformat.write(nb, path)
    print("saved after cell", cell_index, flush=True)


client = NotebookClient(nb, timeout=-1, kernel_name="python3", resources={"metadata": {"path": str(__import__("pathlib").Path(path).parent)}})
client.on_cell_executed = save
try:
    client.execute()
finally:
    nbformat.write(nb, path)
print("notebook finished", flush=True)
