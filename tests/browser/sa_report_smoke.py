"""Check SA SHAP note spacing and interactive plots in an exported report.

Run against a local CEDOS server:
    .venv/Scripts/python tests/browser/sa_report_smoke.py http://127.0.0.1:5094

The test supplies representative SA results and a local Plotly script, so it
does not depend on a model run or the Plotly CDN.
"""

from pathlib import Path
import sys

from plotly.offline.offline import get_plotlyjs
from playwright.sync_api import sync_playwright


BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5002").rstrip("/")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUTPUT = Path(__file__).resolve().parents[2] / ".local" / "sa-report-smoke"


def no_overlap(page, chart_selector):
    assert page.locator(f"{chart_selector} .gtitle").count() == 1
    title = page.locator(f"{chart_selector} .gtitle").bounding_box()
    note = page.locator(f"{chart_selector} .annotation-text").bounding_box()
    assert title and note, "SHAP title or clipped-point note is missing"
    assert note["y"] >= title["y"] + title["height"] + 4, (title, note)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plotly_bundle = OUTPUT / "plotly-2.24.1.min.js"
    plotly_js = plotly_bundle.read_text(encoding="utf-8") if plotly_bundle.exists() else get_plotlyjs()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=CHROME)
        try:
            context = browser.new_context(accept_downloads=True, viewport={"width": 1440, "height": 950})
            context.route("https://cdn.plot.ly/plotly-2.24.1.min.js", lambda route: route.fulfill(
                status=200, content_type="text/javascript", body=plotly_js))
            context.route("https://fonts.googleapis.com/**", lambda route: route.abort())
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            response = page.goto(BASE, wait_until="domcontentloaded", timeout=45000)
            assert response.status == 200
            page.wait_for_function("Boolean(window.cedos) && Boolean(window.Plotly)")
            page.locator('.module-btn[data-module="sa"]').click()

            page.evaluate("""async () => {
                window.currentColumns = ['Feature A', 'Feature B', 'Yield', 'Selectivity'];
                window.columnRoles = {
                    'Feature A': 'feature', 'Feature B': 'feature',
                    Yield: 'objective', Selectivity: 'objective'
                };
                window.currentData = [
                    [1, 2, 10, 20], [2, 3, 20, 30], [3, 4, 30, 40]
                ];
                window.saSelectedFeatures = ['Feature A', 'Feature B'];
                window.saSelectedObjectives = ['Yield', 'Selectivity'];
                const result = {
                    numerical_correlation: {}, num_features: [],
                    extended_obj_names: ['Yield', 'Selectivity'],
                    cat_interactions: {},
                    results: {
                        Yield: {
                            importance: {'Feature A': 60, 'Feature B': 40},
                            shap_data: {
                                features: ['Feature A', 'Feature B'],
                                shap_values: [[-1, 0.2, 0.6], [0.1, -0.3, 0.4]],
                                feature_values: [[1, 2, 3], [2, 3, 4]],
                                display_limit: 0.8,
                                display_clipped_points: 7
                            },
                            fit_flag: 'Good', reliability: 0.84
                        },
                        Selectivity: {
                            importance: {'Feature A': 80, 'Feature B': 20},
                            shap_data: null,
                            fit_flag: 'Limited', reliability: 0.53
                        }
                    }
                };
                window.lastSAResult = result;
                const sa = await import('/static/js/modules/sa.js');
                sa.renderSAResults(result);
            }""")
            page.locator("#sa-shap-active .annotation-text").wait_for()
            page.locator("#sa-shap-active").screenshot(path=str(OUTPUT / "main-shap.png"))
            no_overlap(page, "#sa-shap-active")

            # A script close tag in a feature name must remain data in the report.
            page.evaluate("""() => {
                const source = window.saImpactPlotData.Yield;
                window.saImpactPlotData['Tag </script> target'] = {
                    ...source, objDisplay: 'Tag </script> target'
                };
            }""")
            with page.expect_download() as download_info:
                page.evaluate("""async () => {
                    const sa = await import('/static/js/modules/sa.js');
                    await sa.exportSAReport();
                }""")
            download = download_info.value
            report_path = OUTPUT / "SA_Report.html"
            download.save_as(report_path)
            report_html = report_path.read_text(encoding="utf-8")
            assert "\\u003c/script>" in report_html

            report = context.new_page()
            report_errors = []
            report.on("pageerror", lambda error: report_errors.append(str(error)))
            report.set_content(report_html, wait_until="domcontentloaded")
            report.locator("#report-impact-shap .annotation-text").wait_for()
            assert report.locator("#report-impact-obj option").all_text_contents() == [
                "Yield", "Selectivity", "Tag </script> target"
            ]
            no_overlap(report, "#report-impact-shap")
            report.locator("#report-impact-shap").screenshot(path=str(OUTPUT / "report-shap.png"))

            report.locator("#report-impact-obj").select_option("Selectivity")
            assert report.locator("#report-impact-heading").inner_text() == "Feature Impact: Selectivity"
            assert report.evaluate("document.getElementById('report-impact-bar').data[0].x") == [20, 80]
            assert "No SHAP data available" in report.locator("#report-impact-shap").inner_text()
            report.locator("#report-impact-obj").select_option("Yield")
            report.locator("#report-impact-shap .annotation-text").wait_for()
            no_overlap(report, "#report-impact-shap")
            assert report.evaluate("document.getElementById('report-impact-bar').data[0].x") == [40, 60]
            assert not errors, errors
            assert not report_errors, report_errors

            # A failed Plotly download should explain why the report has no plots.
            offline = context.new_page()
            offline.route("https://cdn.plot.ly/plotly-2.24.1.min.js", lambda route: route.abort())
            offline.set_content(report_html, wait_until="domcontentloaded")
            assert offline.locator("#plotly-unavailable").is_visible()
            print(f"SA main/report SHAP spacing and report objective selection verified; screenshots in {OUTPUT}")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
