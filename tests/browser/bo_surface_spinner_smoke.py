"""Browser regression for BO plot progress cleanup.

Run with a local CEDOS server, e.g.:
    python tests/browser/bo_surface_spinner_smoke.py http://127.0.0.1:5093

The surface job and Plotly renderer are mocked so the test isolates the UI
transition from a completed background job to an interactive chart.
"""

import json
import sys

from playwright.sync_api import sync_playwright


BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5002").rstrip("/")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=CHROME)
        try:
            page = browser.new_page()
            page_errors = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.route("**/bo_surface_submit", lambda route: route.fulfill(
                status=202, content_type="application/json",
                body=json.dumps({"job_id": "bo-surface-smoke", "status": "queued"}),
            ))
            page.route("**/job_status/bo-surface-smoke", lambda route: route.fulfill(
                status=200, content_type="application/json",
                body=json.dumps({"status": "completed", "phase": "Completed", "percent": 100,
                                 "result": {"x": [0, 1], "y": [0, 1],
                                            "z": [[0, 1], [1, 2]],
                                            "color": [[0, 1], [1, 2]],
                                            "x_feat": "x1", "y_feat": "x2",
                                            "z_axis": "yield", "color_obj": "yield",
                                            "experiments": []}}),
            ))

            response = page.goto(BASE, wait_until="domcontentloaded", timeout=45000)
            assert response.status == 200
            page.wait_for_function("Boolean(window.cedos)")
            page.locator("#file-input").set_input_files({
                "name": "surface.csv", "mimeType": "text/csv",
                "buffer": b"x1,x2,yield\n0,0,0\n1,0,1\n0,1,1\n1,1,2\n",
            })
            page.wait_for_function("window.currentData?.length === 4")

            # Plotly.purge deliberately leaves arbitrary progress nodes in place.
            # Before the fix, the Completed spinner remained beside the chart.
            page.evaluate("""() => {
                window.Plotly = {
                    purge() {},
                    newPlot(target) {
                        const element = typeof target === 'string'
                            ? document.getElementById(target) : target;
                        const marker = document.createElement('div');
                        marker.dataset.testChart = 'ready';
                        element.appendChild(marker);
                        return Promise.resolve();
                    },
                    Plots: { resize() {} },
                };
            }""")
            page.locator('.module-btn[data-module="bo"]').click()

            for view, plot_id, control_id in [
                ("contour", "bo-contour-plot", "bo-contour-x"),
                ("surface", "bo-surface-plot", "bo-surf-color"),
            ]:
                page.locator("#bo-vis-type-select").select_option(view)
                page.locator(f"#{plot_id} [data-test-chart=ready]").wait_for(
                    state="attached", timeout=20000)
                assert page.locator(f"#{plot_id} .spinner").count() == 0
                assert "Completed" not in page.locator(f"#{plot_id}").inner_text()
                assert page.locator("#bo-visualization-section").is_visible()
                assert page.locator(f"#{control_id}").is_enabled()

            assert not page_errors, page_errors
            print("BO contour and 3D surface replace the Completed spinner with usable charts.")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
