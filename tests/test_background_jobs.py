import threading
import time

from src.api import jobs
from src.api import routes
from app import app as flask_app


def test_background_job_lifecycle(monkeypatch):
    monkeypatch.setattr(jobs, '_JOBS', {})
    done = threading.Event()

    def worker(job_id, _payload):
        jobs.update_background_job(
            job_id, status='completed', phase='Completed', percent=100,
            result={'ok': True}, finished_at=time.time(),
        )
        done.set()

    job_id = jobs.queue_background_job('test', {}, worker, 'Queued')
    assert job_id
    assert done.wait(2)
    assert jobs.get_background_job_response(job_id)['result'] == {'ok': True}


def test_background_job_cancellation_sets_worker_event(monkeypatch):
    monkeypatch.setattr(jobs, '_JOBS', {})
    started = threading.Event()
    release = threading.Event()

    def worker(_job_id, _payload):
        started.set()
        release.wait(2)

    job_id = jobs.queue_background_job('test', {}, worker, 'Queued')
    assert started.wait(2)
    assert jobs.cancel_background_job(job_id)['status'] == 'cancelling'
    assert jobs.get_cancel_event(job_id).is_set()
    release.set()
    for _ in range(50):
        if jobs.get_background_job_response(job_id)['status'] == 'failed':
            break
        time.sleep(0.01)


def test_stopped_supervisor_cannot_leave_an_active_slot(monkeypatch):
    monkeypatch.setattr(jobs, '_JOBS', {})
    returned = threading.Event()

    def worker(_job_id, _payload):
        returned.set()

    job_id = jobs.queue_background_job('test', {}, worker, 'Queued')
    assert returned.wait(2)
    for _ in range(20):
        response = jobs.get_background_job_response(job_id)
        if response['status'] == 'failed':
            break
        time.sleep(0.01)
    assert response['status'] == 'failed'
    assert response['error_code'] == 'supervisor_exit'


def test_bo_surface_submit_rejects_rows_above_surface_limit():
    flask_app.config['TESTING'] = True
    payload = {
        'data': [[0, 0, 0]] * 651,
        'columns': ['x', 'y', 'objective'],
        'features': [
            {'name': 'x', 'type': 'continuous', 'range': '[0, 1]'},
            {'name': 'y', 'type': 'continuous', 'range': '[0, 1]'},
        ],
        'objectives': [{'name': 'objective', 'type': 'maximize'}],
    }
    with flask_app.test_client() as client:
        response = client.post('/bo_surface_submit', json=payload)
    assert response.status_code == 400
    assert 'limited to 650' in response.get_json()['error']


def _bo_payload(batch_size=2):
    return {
        'data': [[0.1, 1.0], [0.5, 2.0], [0.9, 1.5]],
        'columns': ['x', 'objective'],
        'features': [
            {'name': 'x', 'type': 'continuous', 'range': '[0, 1]'},
        ],
        'objectives': [
            {'name': 'objective', 'type': 'maximize', 'importance': 1},
        ],
        'tweaks': {'batch_size': batch_size, 'constraints': {}},
    }


def test_bo_submit_validates_then_returns_immediately(monkeypatch):
    captured = {}

    def queue(kind, payload, target, phase):
        captured.update(
            kind=kind, payload=payload, target=target, phase=phase
        )
        return 'bo-job-id'

    monkeypatch.setattr(routes, '_queue_idempotent_background_job', queue)
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        response = client.post('/bo_submit', json=_bo_payload())
    assert response.status_code == 202
    assert response.get_json()['job_id'] == 'bo-job-id'
    assert captured['kind'] == 'bo-optimization'
    assert captured['target'] is routes._run_bo_job


def test_bo_submit_reports_shared_analysis_capacity(monkeypatch):
    monkeypatch.setattr(routes, '_queue_idempotent_background_job', lambda *_args: None)
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        response = client.post('/bo_submit', json=_bo_payload())
    assert response.status_code == 429
    assert response.get_json()['error_code'] == 'busy'


def test_bo_submit_rejects_oversized_batch_before_queue(monkeypatch):
    queued = False

    def queue(*_args):
        nonlocal queued
        queued = True

    monkeypatch.setattr(routes, '_queue_idempotent_background_job', queue)
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        response = client.post('/bo_submit', json=_bo_payload(batch_size=21))
    assert response.status_code == 400
    assert queued is False


def test_bo_submit_rejects_unknown_large_data_mode_before_queue(monkeypatch):
    queued = False

    def queue(*_args):
        nonlocal queued
        queued = True

    payload = _bo_payload()
    payload['tweaks']['data_mode'] = 'silent-truncation'
    monkeypatch.setattr(routes, '_queue_idempotent_background_job', queue)
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        response = client.post('/bo_submit', json=payload)

    assert response.status_code == 400
    assert 'Unknown BO data mode' in response.get_json()['error']
    assert queued is False


def test_bo_submit_rejects_nparego_with_unequal_importance(monkeypatch):
    queued = False

    def queue(*_args):
        nonlocal queued
        queued = True

    payload = _bo_payload()
    payload['objectives'] = [
        {'name': 'objective', 'type': 'maximize', 'importance': 70},
        {'name': 'objective_2', 'type': 'maximize', 'importance': 30},
    ]
    payload['columns'].append('objective_2')
    payload['data'] = [row + [row[-1] * 2] for row in payload['data']]
    payload['tweaks']['multi_objective_method'] = 'qLogNParEGO'
    monkeypatch.setattr(routes, '_queue_idempotent_background_job', queue)
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        response = client.post('/bo_submit', json=payload)

    assert response.status_code == 400
    assert 'unavailable' in response.get_json()['error']
    assert queued is False


def test_hosted_request_cannot_enable_unsafe_full_override(monkeypatch):
    captured = {}

    def queue(_kind, payload, _target, _phase):
        captured.update(payload)
        return 'bo-job-id'

    payload = _bo_payload()
    payload['tweaks']['_allow_unsafe_full'] = True
    monkeypatch.setenv('SPACE_ID', 'realgcp/TransEDOS')
    monkeypatch.setattr(routes, '_queue_idempotent_background_job', queue)
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        response = client.post('/bo_submit', json=payload)

    assert response.status_code == 202
    assert captured['tweaks']['_allow_unsafe_full'] is False
