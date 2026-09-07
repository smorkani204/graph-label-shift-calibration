import json

from src.utils.results import save_json_results


def test_save_json_results(tmp_path):
    results = {
        "experiment": "test",
        "value": 0.123,
        "valid": True,
    }

    output_path = tmp_path / "results.json"

    save_json_results(
        results,
        output_path,
    )

    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)

    assert loaded == results