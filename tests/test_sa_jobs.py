import threading
import time
import unittest
from unittest.mock import patch

from app import app
from src.api import jobs, routes
from src.core.sa import AnalysisCancelled, _check_cancel
from src.core.transfer_bo import (
    TransferCalculationCancelled,
    TransferGPFitError,
    TransferGPFitTimeout,
    _transfer_control_check,
)


class StatisticalAnalysisCancellationTests(unittest.TestCase):
    def setUp(self):
        with jobs._JOBS_LOCK:
            jobs._JOBS.clear()
        with routes._REQUEST_JOB_IDS_LOCK:
            routes._REQUEST_JOB_IDS.clear()

    def test_duplicate_submission_id_returns_original_job(self):
        blocker = threading.Event()

        def worker(_job_id, _payload):
            blocker.wait(2)

        headers = {'X-Analysis-Request-ID': 'same-browser-submission'}
        with app.test_request_context(headers=headers):
            first_id = routes._queue_idempotent_background_job(
                'test', {}, worker, 'Queued'
            )
        with app.test_request_context(headers=headers):
            second_id = routes._queue_idempotent_background_job(
                'test', {}, worker, 'Queued'
            )

        self.assertEqual(second_id, first_id)
        with jobs._JOBS_LOCK:
            self.assertEqual(len(jobs._JOBS), 1)
        blocker.set()

    def test_cancel_event_interrupts_at_cooperative_checkpoint(self):
        event = threading.Event()
        _check_cancel(event)
        event.set()
        with self.assertRaises(AnalysisCancelled):
            _check_cancel(event)

    def test_transfer_optimization_runs_as_background_job(self):
        payload = {
            'data': [[0.0, 'A', 1.0], [1.0, 'B', 2.0]],
            'columns': ['X', 'Prior', 'Y'],
            'features': [{'name': 'X', 'type': 'continuous', 'range': '[0, 1]'}],
            'objectives': [{'name': 'Y', 'type': 'maximize', 'importance': 100}],
            'priors': [{'name': 'Prior', 'col_type': 'categorical', 'target_value': 'A'}],
            'tweaks': {'max_gp_rows': 100},
        }
        mocked = (
            [{'X': 0.5}],
            {'Y': {'r2': 0.8, 'mse': 0.1}},
            {'model_rows_used': 2},
            {'constrained': False},
        )
        with app.test_client() as client, patch(
            'src.api.routes.run_transfer_bo_optimization', return_value=mocked
        ):
            response = client.post('/transfer_optimize_submit', json=payload)
            self.assertEqual(response.status_code, 202, response.get_data(as_text=True))
            job_id = response.get_json()['job_id']
            for _ in range(100):
                status_response = client.get(f'/job_status/{job_id}')
                status = status_response.get_json()
                if status['status'] == 'completed':
                    self.assertEqual(status['result']['suggestions'][0]['X'], 0.5)
                    break
                time.sleep(0.01)
            else:
                self.fail(f'Background job did not complete: {status}')

    def test_transfer_selection_preview_reports_actual_rows(self):
        payload = {
            'data': [[float(i), 'A' if i < 6 else 'B', float(i)] for i in range(10)],
            'columns': ['X', 'Prior', 'Y'],
            'features': [{'name': 'X', 'type': 'continuous', 'range': '[0, 9]'}],
            'objectives': [{'name': 'Y', 'type': 'maximize', 'importance': 100}],
            'priors': [{'name': 'Prior', 'col_type': 'categorical', 'target_value': 'A', 'prior_weight': 100}],
            'prior_combine': 'average',
            'tweaks': {'max_gp_rows': 4},
        }
        with app.test_client() as client:
            response = client.post('/transfer_selection_preview', json=payload)
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        result = response.get_json()
        self.assertEqual(result['eligible_rows'], 6)
        self.assertEqual(result['rows_to_fit'], 4)

    def _wait_for_completion(self, client, job_id):
        for _ in range(100):
            response = client.get(f'/job_status/{job_id}')
            status = response.get_json()
            if status['status'] == 'completed':
                return status['result']
            if status['status'] == 'failed':
                self.fail(f'Background job failed: {status}')
            time.sleep(0.01)
        self.fail(f'Background job did not complete: {status}')

    def test_transfer_estimator_runs_as_background_job(self):
        payload = {
            'data': [[0.0, 'A', 1.0], [1.0, 'B', 2.0]],
            'columns': ['X', 'Prior', 'Y'],
            'inputs': {'X': 0.5, 'Prior': 'A'},
            'features': [{'name': 'X', 'type': 'continuous', 'range': '[0, 1]'}],
            'objectives': [{'name': 'Y', 'type': 'maximize', 'importance': 100}],
            'priors': [{'name': 'Prior', 'col_type': 'categorical', 'target_value': 'A'}],
            'tweaks': {'max_gp_rows': 100},
        }
        mocked = ({'Y': {'mean': 1.5, 'std': 0.1, 'success': 50.0}}, 50.0, [], {'model_rows_used': 2})
        with app.test_client() as client, patch(
            'src.api.routes.run_transfer_bo_estimation', return_value=mocked
        ):
            response = client.post('/transfer_estimate_submit', json=payload)
            self.assertEqual(response.status_code, 202, response.get_data(as_text=True))
            result = self._wait_for_completion(client, response.get_json()['job_id'])
            self.assertEqual(result['results']['Y']['mean'], 1.5)
            self.assertEqual(result['workload']['model_rows_used'], 2)

    def test_sa_estimator_runs_as_background_job(self):
        payload = {
            'data': [[0.0, 1.0], [1.0, 2.0]],
            'columns': ['X', 'Y'],
            'inputs': {'X': 0.5},
            'features': [{'name': 'X', 'type': 'continuous'}],
            'objectives': [{'name': 'Y', 'type': 'maximize', 'importance': 100}],
            'model': 'mlp',
            'params': {'layers': [8], 'mlp_iter': 50},
            'tweaks': {},
        }
        mocked = ({'Y': 1.5}, {'Y': 0.1}, {'Y': 50.0}, 50.0, [])
        with app.test_client() as client, patch(
            'src.api.routes.estimate_sa', return_value=mocked
        ):
            response = client.post('/sa_estimate_submit', json=payload)
            self.assertEqual(response.status_code, 202, response.get_data(as_text=True))
            result = self._wait_for_completion(client, response.get_json()['job_id'])
            self.assertEqual(result['predictions']['Y'], 1.5)
            self.assertEqual(result['global_success'], 50.0)


class TransferReducedRowRecoveryTests(unittest.TestCase):
    def test_cancel_watchdog_releases_a_stuck_job_slot(self):
        blocker = threading.Event()

        def stuck_worker(_job_id, _payload):
            blocker.wait(2)

        job_id = jobs.queue_background_job('stuck', {}, stuck_worker, 'Queued')
        self.assertIsNotNone(job_id)
        jobs.update_background_job(job_id, status='cancelling')

        routes._release_job_slot_after_grace(job_id, grace_seconds=0)

        status = jobs.get_background_job_response(job_id)
        self.assertEqual(status['status'], 'failed')
        self.assertIn('slot was released', status['error'])
        def complete_replacement(replacement_id, _payload):
            jobs.update_background_job(
                replacement_id,
                status='completed',
                phase='Completed',
                percent=100,
                result={},
                finished_at=time.time(),
            )

        replacement = jobs.queue_background_job(
            'replacement', {}, complete_replacement, 'Queued'
        )
        self.assertIsNotNone(replacement)
        blocker.set()

    def test_transfer_control_honors_timeout_and_cancellation(self):
        with self.assertRaises(TransferGPFitTimeout):
            _transfer_control_check(deadline_monotonic=time.monotonic() - 1)

        cancel_event = threading.Event()
        cancel_event.set()
        with self.assertRaises(TransferCalculationCancelled):
            _transfer_control_check(cancel_event=cancel_event)

    def test_timeout_retries_with_fewer_rows(self):
        attempted_limits = []

        def operation(row_limit):
            attempted_limits.append(row_limit)
            if len(attempted_limits) == 1:
                raise TransferGPFitTimeout('synthetic attempt timeout')
            return {'row_limit': row_limit}

        result, recovery = routes._run_transfer_with_row_recovery(
            operation, requested_rows=1800, enabled=True
        )

        self.assertEqual(attempted_limits, [1800, 900])
        self.assertEqual(result['row_limit'], 900)
        self.assertTrue(recovery['used_reduced_rows'])

    def test_recovery_is_opt_in_and_reports_the_reduced_limit(self):
        attempted_limits = []

        def operation(row_limit):
            attempted_limits.append(row_limit)
            if len(attempted_limits) == 1:
                raise TransferGPFitError('synthetic fitting failure')
            return {'row_limit': row_limit}

        result, recovery = routes._run_transfer_with_row_recovery(
            operation, requested_rows=800, enabled=True
        )

        self.assertEqual(attempted_limits, [800, 400])
        self.assertEqual(result['row_limit'], 400)
        self.assertTrue(recovery['used_reduced_rows'])
        self.assertEqual(recovery['requested_gp_rows'], 800)
        self.assertEqual(recovery['retry_max_gp_rows'], 400)

    def test_recovery_reduces_effective_rows_instead_of_raw_dataset_cap(self):
        attempted_limits = []

        def operation(row_limit):
            attempted_limits.append(row_limit)
            if len(attempted_limits) == 1:
                raise TransferGPFitTimeout('synthetic attempt timeout')
            return {'row_limit': row_limit}

        result, recovery = routes._run_transfer_with_row_recovery(
            operation,
            requested_rows=27497,
            enabled=True,
            resolve_effective_rows=lambda: 1683,
        )

        self.assertEqual(attempted_limits, [27497, 842])
        self.assertEqual(result['row_limit'], 842)
        self.assertEqual(recovery['configured_max_gp_rows'], 27497)
        self.assertEqual(recovery['requested_gp_rows'], 1683)
        self.assertEqual(recovery['retry_max_gp_rows'], 842)
        self.assertEqual(recovery['attempted_limits'], [1683, 842])

        workload = routes._with_recovery_metadata(
            {'requested_gp_rows': 842, 'model_rows_used': 842}, recovery
        )
        self.assertEqual(workload['requested_gp_rows'], 27497)
        self.assertEqual(workload['effective_requested_gp_rows'], 1683)

    def test_effective_row_preview_is_lazy_when_initial_attempt_succeeds(self):
        resolver_called = False

        def resolver():
            nonlocal resolver_called
            resolver_called = True
            return 100

        result, recovery = routes._run_transfer_with_row_recovery(
            lambda row_limit: {'row_limit': row_limit},
            requested_rows=1000,
            enabled=True,
            resolve_effective_rows=resolver,
        )

        self.assertEqual(result['row_limit'], 1000)
        self.assertIsNone(recovery)
        self.assertFalse(resolver_called)

    def test_recovery_disabled_preserves_the_original_failure(self):
        def operation(_row_limit):
            raise TransferGPFitError('synthetic fitting failure')

        with self.assertRaises(TransferGPFitError):
            routes._run_transfer_with_row_recovery(
                operation, requested_rows=800, enabled=False
            )

    def test_recovery_retry_drops_an_exact_row_reuse_request(self):
        tweaks = {
            'max_gp_rows': 800,
            'reuse_training_row_indices': [2, 5, 8],
        }
        first_attempt = routes._transfer_tweaks_for_attempt(tweaks, 800)
        retry_attempt = routes._transfer_tweaks_for_attempt(tweaks, 400)

        self.assertEqual(first_attempt['reuse_training_row_indices'], [2, 5, 8])
        self.assertNotIn('reuse_training_row_indices', retry_attempt)


if __name__ == '__main__':
    unittest.main()
