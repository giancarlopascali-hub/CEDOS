"""Check the CEDOS Prior/Feature role transition in a live browser.

Run against a local server: python tests/browser/role_transition_smoke.py URL
"""

import sys

from playwright.sync_api import sync_playwright


BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5002").rstrip("/")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=CHROME)
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        response = page.goto(BASE, wait_until="domcontentloaded", timeout=45000)
        assert response.status == 200
        page.wait_for_function("Boolean(window.cedos)")
        page.locator("#file-input").set_input_files({
            "name": "roles.csv", "mimeType": "text/csv",
            "buffer": b"temperature,context,y\n20,A,1\n30,B,2\n40,A,3\n",
        })
        page.wait_for_function("window.currentData?.length === 3")
        data_before = page.evaluate("JSON.stringify(window.currentData)")

        page.locator("#cedos-transbo-toggle").check()
        page.locator('.module-btn[data-module="tbo"]').click()
        role = page.locator("#main-table-head th:nth-child(2) .type-toggle")
        role.click()  # Feature to Objective.
        role.click()  # Objective to Prior.
        page.wait_for_function("window.columnRoles.context === 'prior'")
        page.locator('#tbo-module .tab-btn[data-tab="tbo-priors-tab"]').click()
        page.locator('.tbo-prior-cat-select[data-col="context"]').select_option("A")
        assert page.evaluate("window.priorConfigs.context.target_value") == "A"
        objective_before = page.evaluate("JSON.stringify(window.objectiveConfigs.y)")

        page.locator("#cedos-transbo-toggle").uncheck()
        assert page.evaluate("window.currentModule") == "bo"
        assert page.evaluate("window.columnRoles.context") == "feature"
        bo_feature = page.evaluate("window.boFeatureConfigs.context")
        assert bo_feature["type"] == "categorical", bo_feature
        assert bo_feature["choices"] == ["A", "B"], bo_feature
        assert page.evaluate("window.saFeatureConfigs.context.type") == "categorical"
        assert page.evaluate("window.priorConfigs.context === undefined")
        assert page.evaluate("JSON.stringify(window.currentData)") == data_before
        assert page.evaluate("JSON.stringify(window.objectiveConfigs.y)") == objective_before

        page.locator("#cedos-transbo-toggle").check()
        assert page.evaluate("window.columnRoles.context") == "prior"
        assert page.evaluate("window.priorConfigs.context.target_value") == "A"

        # A BO configuration edit is deliberate and should keep the column F.
        page.locator("#cedos-transbo-toggle").uncheck()
        range_input = page.locator('.bo-f-range[data-col="context"]')
        range_input.fill("A, B, C")
        range_input.press("Tab")
        assert page.evaluate("window.boFeatureConfigs.context.range") == "A, B, C"
        page.locator("#cedos-transbo-toggle").check()
        assert page.evaluate("window.columnRoles.context") == "feature"
        assert page.evaluate("window.boFeatureConfigs.context.range") == "A, B, C"

        # An explicit role change in BO should also remain untouched.
        role.click()  # Feature to Objective in TransBO.
        role.click()  # Objective to Prior in TransBO.
        page.wait_for_function("window.columnRoles.context === 'prior'")
        page.locator("#cedos-transbo-toggle").uncheck()
        role.click()  # Feature to Objective in BO.
        assert page.evaluate("window.columnRoles.context") == "objective"
        page.locator("#cedos-transbo-toggle").check()
        assert page.evaluate("window.columnRoles.context") == "objective"
        assert not errors, errors
        browser.close()
    print("PASS: Prior columns become BO Features and restore only when untouched")


if __name__ == "__main__":
    main()
