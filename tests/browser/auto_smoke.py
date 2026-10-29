"""Run against a local CEDOS server: python tests/browser/auto_smoke.py URL.

The optimizer jobs are mocked; the identified CEDOS/IMEX bridge is real. Chrome
is required, but no hardware or configured IMEX folder is needed.
"""

import json
import sys

from playwright.sync_api import sync_playwright


BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5091").rstrip("/")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def make_page(browser):
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    jobs = {}

    def submit_job(route):
        body = route.request.post_data_json
        features = body["features"]
        suggestion = {}
        for feature in features:
            suggestion[feature["name"]] = (
                feature.get("choices", ["A"])[0]
                if feature.get("type") == "categorical" else 4
            )
        job_id = "smoke-" + str(len(jobs) + 1)
        jobs[job_id] = suggestion
        route.fulfill(status=200, content_type="application/json", body=json.dumps({
            "job_id": job_id, "status": "queued"
        }))

    def job_status(route):
        job_id = route.request.url.rsplit("/", 1)[-1]
        route.fulfill(status=200, content_type="application/json", body=json.dumps({
            "status": "completed", "result": {
                "suggestions": [jobs[job_id]], "constraint_status": {}
            }
        }))

    page.route("**/bo_submit", submit_job)
    page.route("**/transfer_optimize_submit", submit_job)
    page.route("**/job_status/smoke-*", job_status)
    response = page.goto(BASE, wait_until="domcontentloaded", timeout=45000)
    assert response.status == 200
    page.wait_for_function("Boolean(window.cedos && window.initAutoModule)")
    return page, errors


def configure_existing_data(page, transfer=False):
    page.evaluate("""transfer => {
        window.currentColumns = transfer ? ['x', 'source', 'y'] : ['x', 'y'];
        window.currentData = transfer
            ? [[1, 'A', 1], [2, 'A', 2], [3, 'A', 3]]
            : [[1, 1], [2, 2], [3, 3]];
        window.columnRoles = transfer
            ? {x:'feature', source:'prior', y:'objective'}
            : {x:'feature', y:'objective'};
        window.boFeatureConfigs = {x: {type:'continuous', range:'0,10', included:true}};
        window.saFeatureConfigs = {x: {type:'continuous', included:true}};
        window.objectiveConfigs = {y: {
            name:'y', type:'maximize', target:'', importance:100, included:true
        }};
        if (transfer) window.priorConfigs = {source: {
            col_type:'categorical', target_value:'B', prior_weight:100
        }};
        window.cedos.notifyDatasetMutation('browser smoke fixture');
        window.cedos.switchModuleSafe(transfer ? 'tbo' : 'bo');
        window.renderSetup();
        window.renderMainTable();
    }""", transfer)


def start_direct_optimizer(page):
    page.locator('.module-btn[data-module="auto"]').click()
    page.locator('#check-imex-started').check()
    page.locator('#auto-select-bo').click()
    page.locator('.module-btn[data-module="auto"]').click()
    page.locator('#auto-goto-bo').click()
    page.locator('.module-btn[data-module="auto"]').click()
    page.locator('#auto-stop-type').select_option('iterations')
    page.locator('#auto-stop-value').fill('1')
    assert page.locator('#execute-autobo-btn').is_visible()
    page.locator('#execute-autobo-btn').click()
    page.wait_for_function("window.cedos.campaignState === 'awaiting_hardware'", timeout=30000)


def send_hardware_result(page, measured=42):
    proposal_response = page.request.get(BASE + '/auto/proposal')
    assert proposal_response.status == 200, proposal_response.status
    proposal = proposal_response.json()
    campaign_id = proposal['campaign_id']
    proposal_id = proposal['proposal_id']
    assert isinstance(proposal['dataset_rev'], int)
    assert isinstance(proposal['config_rev'], int)
    assert len(proposal['rows']) == len(proposal['features_only'])
    acknowledgement = page.request.post(
        BASE + f'/auto/proposal/{proposal_id}/ack',
        data={'campaign_id': campaign_id}
    )
    assert acknowledgement.ok, acknowledgement.text()
    results = {'campaign_id': campaign_id, 'proposal_id': proposal_id, 'rows': [
        {'row_id': row['row_id'], **{name: measured for name in proposal['objective_names']}}
        for row in proposal['rows']
    ]}
    posted = page.request.post(BASE + '/auto/results', data=results)
    assert posted.ok, posted.text()
    return proposal


def run_direct(browser, transfer=False):
    page, errors = make_page(browser)
    auto_guide = page.locator('a[href="/auto-guide"]').first
    back_to_auto = page.locator('#doe-module .back-to-auto-btn')
    assert not auto_guide.is_visible()
    assert not back_to_auto.is_visible()
    page.locator('#cedos-auto-toggle').check()
    assert auto_guide.is_visible()
    assert back_to_auto.is_visible()
    page.locator('#cedos-transbo-toggle').set_checked(transfer)
    assert page.evaluate('window.cedos.activeOptimizer') == ('tbo' if transfer else 'bo')
    assert page.locator('.module-btn[data-module="bo"]').is_visible() != transfer
    assert page.locator('.module-btn[data-module="tbo"]').is_visible() == transfer
    for module in ['doe', 'bo', 'tbo', 'sa']:
        assert page.locator(f'#{module}-setup-download-btn').count() == 1
        assert page.locator(f'#{module}-setup-upload-btn').count() == 1
    configure_existing_data(page, transfer)
    start_direct_optimizer(page)
    proposal = send_hardware_result(page)
    assert proposal['optimizer'] == ('tbo' if transfer else 'bo')
    if transfer:
        assert proposal['rows'][0]['source'] == 'B'
        assert page.evaluate('window.columnRoles.source') == 'prior'
    page.wait_for_function("window.cedos.campaignState === 'completed'", timeout=20000)
    snapshot = page.evaluate('({data:window.currentData, iterations:window.autoCurrentIteration})')
    assert snapshot['iterations'] == 1
    assert snapshot['data'][-1][-1] == 42
    assert not errors, errors
    page.close()
    return 'direct TransBO' if transfer else 'direct BO'


def run_doe_handoff(browser):
    page, errors = make_page(browser)
    page.on('dialog', lambda dialog: dialog.accept())
    page.locator('#cedos-auto-toggle').check()
    page.locator('#cedos-transbo-toggle').check()
    page.route('**/doe', lambda route: route.fulfill(
        status=200, content_type='application/json',
        body=json.dumps({'suggestions': [{'x': 1}], 'metrics': {}})
    ))
    page.locator('#doe-add-feature-btn').click()
    page.locator('#doe-features-list .doe-f-name').fill('x')
    page.locator('#doe-features-list .doe-f-range').fill('0,10')
    page.locator('#doe-objectives-tab').evaluate('(el) => el.classList.remove("hidden")')
    page.locator('#doe-add-objective-btn').click()
    page.locator('#doe-objectives-list .doe-o-name').fill('y')
    page.locator('.module-btn[data-module="auto"]').click()
    page.locator('#check-imex-started').check()
    page.locator('#auto-select-doe').click()
    page.locator('.module-btn[data-module="auto"]').click()
    page.locator('#execute-autodoe-btn').click()
    page.wait_for_function("window.cedos.campaignState === 'awaiting_hardware'", timeout=20000)
    proposal = send_hardware_result(page, measured=12)
    assert proposal['optimizer'] == 'doe'
    page.wait_for_function("window.cedos.campaignState === 'review_optimizer'", timeout=20000)
    assert page.evaluate('window.currentModule') == 'tbo'
    assert page.evaluate('window.currentData[0][1]') == 12
    assert page.evaluate('window.cedos.canImportSetup("tbo")')
    page.evaluate('window.cedos.switchModuleSafe("auto")')
    page.locator('#auto-goto-bo').click()
    page.evaluate('window.cedos.switchModuleSafe("auto")')
    page.locator('#auto-stop-type').select_option('iterations')
    page.locator('#auto-stop-value').fill('1')
    page.locator('#execute-autobo-btn').click()
    page.wait_for_function("window.cedos.campaignState === 'awaiting_hardware'", timeout=30000)
    optimized = send_hardware_result(page, measured=99)
    assert optimized['optimizer'] == 'tbo'
    page.wait_for_function("window.cedos.campaignState === 'completed'", timeout=20000)
    assert page.evaluate('window.autoCurrentIteration') == 1
    assert page.evaluate('window.currentData[1][1]') == 99
    assert not errors, errors
    page.close()
    return 'DoE to TransBO review and measured optimization'


def run_reload_recovery(browser):
    page, errors = make_page(browser)
    page.on('dialog', lambda dialog: dialog.accept())
    page.locator('#cedos-auto-toggle').check()
    configure_existing_data(page)
    start_direct_optimizer(page)
    old_id = page.evaluate('sessionStorage.getItem("cedosAutoCampaignId")')
    assert old_id
    page.reload(wait_until='domcontentloaded')
    page.locator('#cedos-auto-toggle').check()
    page.locator('.module-btn[data-module="auto"]').click()
    page.locator('#auto-recover-btn').wait_for(timeout=15000)
    page.locator('#auto-recover-btn').click()
    page.wait_for_function('!document.getElementById("auto-recover-btn")')
    status = page.request.get(BASE + '/auto/status', params={'campaign_id': old_id}).json()
    assert status['status'] == 'cancelled'
    assert not errors, errors
    page.close()
    return 'reload recovery releases orphaned proposal'


if __name__ == '__main__':
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=CHROME)
        try:
            for result in [run_direct(browser), run_direct(browser, True),
                           run_doe_handoff(browser), run_reload_recovery(browser)]:
                print('PASS:', result)
        finally:
            browser.close()
