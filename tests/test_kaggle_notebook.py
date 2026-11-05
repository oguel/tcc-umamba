"""Notebook distribuído não inicia campanha por executar todas as células."""

import ast
import json
from pathlib import Path


def test_notebook_is_clean_and_training_disabled_by_default():
    path = Path(__file__).resolve().parents[1] / "notebooks/13_kaggle_training.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = []
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert notebook["cells"][i - 1]["cell_type"] == "markdown"
            assert not cell["outputs"]
            assert cell["execution_count"] is None
            text = "".join(cell["source"])
            ast.parse(text)
            code.append(text)
    first = ast.parse(code[0])
    values = {
        node.targets[0].id: node.value.value
        for node in first.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert values["START_TRAINING"] is False
    assert values["SETUP_MAMBA"] is False
    assert values["EPOCHS"] == 100
