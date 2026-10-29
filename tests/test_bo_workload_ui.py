import json
import pathlib
import shutil
import subprocess

import pytest


NODE_AVAILABLE = shutil.which("node") is not None


@pytest.mark.skipif(not NODE_AVAILABLE, reason="Node.js is not installed")
def test_browser_workload_preview_matches_policy_boundaries():
    module_uri = (
        pathlib.Path(__file__).parents[1]
        / "static/js/modules/bo_workload.js"
    ).resolve().as_uri()
    script = f"""
        import {{ estimateBOWorkload, recommendedMultiObjectiveMethod }} from '{module_uri}';
        const result = {{
            equal: recommendedMultiObjectiveMethod(4, [25, 25, 25, 25]),
            unequal: recommendedMultiObjectiveMethod(4, [40, 30, 20, 10]),
            ordinary: estimateBOWorkload({{
                rowCount: 69, featureCount: 16, objectiveCount: 7,
                batchSize: 10, method: 'qLogNParEGO'
            }}),
            large: estimateBOWorkload({{
                rowCount: 10000, featureCount: 16, objectiveCount: 10,
                batchSize: 10, method: 'qLogNParEGO'
            }})
        }};
        console.log(JSON.stringify(result));
    """
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["equal"] == "qLogNParEGO"
    assert result["unequal"] == "scalarized-qLogEI"
    assert result["ordinary"]["controlsRequired"] is False
    assert result["large"]["controlsRequired"] is True
    assert 20 <= result["large"]["recommendedRows"] <= 250
