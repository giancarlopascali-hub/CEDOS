import pathlib
import threading
import time
import unittest
from unittest.mock import patch


class SurfaceContractTests(unittest.TestCase):
    def test_transfer_surface_source_has_independent_height_and_color_outputs(self):
        # This guards the response contract without fitting an expensive GP in a
        # unit test; integration smoke tests exercise the model path separately.
        from src.core import transfer_bo

        names = transfer_bo.run_transfer_bo_surface.__code__.co_names
        constants = transfer_bo.run_transfer_bo_surface.__code__.co_consts
        source_markers = repr((names, constants))
        self.assertIn('color', source_markers)
        self.assertIn('color_label', source_markers)
        self.assertIn('z_label', source_markers)
        self.assertIn('experiments', source_markers)
        self.assertIn('overlay', source_markers)

    def test_transfer_surface_honors_pre_cancelled_work(self):
        from src.core.bo import SurfaceCalculationCancelled
        from src.core.transfer_bo import run_transfer_bo_surface

        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(SurfaceCalculationCancelled):
            run_transfer_bo_surface(
                df=None,
                features_config=[],
                objectives_config=[],
                priors_config=[],
                tweaks={},
                x_feat=None,
                y_feat=None,
                cancel_event=cancelled,
            )

    def test_browser_uses_cancellable_surface_jobs_and_slice_overlays(self):
        source = pathlib.Path('static/js/modules/tbo.js').read_text(encoding='utf-8')
        bo_source = pathlib.Path('static/js/modules/bo.js').read_text(encoding='utf-8')
        utils_source = pathlib.Path('static/js/utils.js').read_text(encoding='utf-8')
        self.assertIn("error.name === 'AbortError'", source)
        self.assertIn('cancelActiveTBOPlot', source)
        self.assertIn('cancelActiveBOPlot', bo_source)
        self.assertIn('/transfer_surface_submit', source)
        self.assertIn('hasConfiguredTBOPriorTarget', source)
        self.assertIn('BO_SURFACE_MAX_ROWS', bo_source)
        self.assertIn('buildTBOExperimentTraces', source)
        self.assertIn('surfaceObjectives.map', source)
        self.assertIn('waitForBackgroundJobTermination', utils_source)
        self.assertIn('waitForTerminal: true', source)
        self.assertIn('waitForTerminal: true', bo_source)

    def test_background_polling_retries_gateway_failures(self):
        source = pathlib.Path('static/js/utils.js').read_text(encoding='utf-8')
        polling_block = source[source.index('export async function submitAndPollJob'):source.index('export async function waitForBackgroundJobTermination')]
        self.assertIn('[502, 503, 504]', polling_block)
        self.assertIn('retryAfterInterruption', polling_block)

    def test_transfer_surface_job_cancels_server_work(self):
        from app import app
        from src.api import routes
        from src.core.bo import SurfaceCalculationCancelled

        started = threading.Event()

        def cancellable_surface(**kwargs):
            started.set()
            while not kwargs['cancel_event'].wait(0.01):
                pass
            raise SurfaceCalculationCancelled('Response-surface calculation cancelled by user.')

        payload = {
            'data': [[0.0, 0.0, 50.0]],
            'columns': ['X', 'Y', 'Yield'],
            'features': [
                {'name': 'X', 'type': 'continuous', 'range': '[0, 1]'},
                {'name': 'Y', 'type': 'continuous', 'range': '[0, 1]'},
            ],
            'objectives': [{'name': 'Yield', 'type': 'maximize'}],
            'priors': [{'name': 'X', 'col_type': 'numerical', 'target_value': 0.5}],
            'tweaks': {'max_gp_rows': 3},
            'x_feat': 'X',
            'y_feat': 'Y',
            'color_obj': 'Yield',
        }

        with patch.object(routes, 'run_transfer_bo_surface', side_effect=cancellable_surface):
            with app.test_client() as client:
                response = client.post('/transfer_surface_submit', json=payload)
                self.assertEqual(response.status_code, 202)
                job_id = response.get_json()['job_id']
                self.assertTrue(started.wait(1))
                self.assertEqual(client.post(f'/job_cancel/{job_id}').status_code, 200)

                for _ in range(100):
                    status = client.get(f'/job_status/{job_id}').get_json()
                    if status['status'] == 'cancelled':
                        break
                    time.sleep(0.01)
                self.assertEqual(status['status'], 'cancelled')


    def test_transfer_surface_requires_a_configured_prior_target(self):
        from app import app

        payload = {
            'data': [[0.0, 0.0, 50.0]],
            'columns': ['X', 'Y', 'Yield'],
            'features': [
                {'name': 'X', 'type': 'continuous', 'range': '[0, 1]'},
                {'name': 'Y', 'type': 'continuous', 'range': '[0, 1]'},
            ],
            'objectives': [{'name': 'Yield', 'type': 'maximize'}],
            'priors': [],
        }
        with app.test_client() as client:
            response = client.post('/transfer_surface_submit', json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn('Prior target', response.get_json()['error'])

    def test_bo_surface_rejects_oversized_dataset_before_queueing(self):
        from app import app
        from src.api.routes import BO_SURFACE_MAX_ROWS

        payload = {
            'data': [[0.0, 0.0, 50.0]] * (BO_SURFACE_MAX_ROWS + 1),
            'columns': ['X', 'Y', 'Yield'],
            'features': [
                {'name': 'X', 'type': 'continuous', 'range': '[0, 1]'},
                {'name': 'Y', 'type': 'continuous', 'range': '[0, 1]'},
            ],
            'objectives': [{'name': 'Yield', 'type': 'maximize'}],
        }
        with app.test_client() as client:
            response = client.post('/bo_surface_submit', json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn('limited', response.get_json()['error'])

    def test_bo_surface_job_cancels_server_work(self):
        from app import app
        from src.api import routes
        from src.core.bo import SurfaceCalculationCancelled

        started = threading.Event()

        def cancellable_surface(**kwargs):
            started.set()
            while not kwargs['cancel_event'].wait(0.01):
                pass
            raise SurfaceCalculationCancelled('Response-surface calculation cancelled by user.')

        payload = {
            'data': [[0.0, 0.0, 50.0]],
            'columns': ['X', 'Y', 'Yield'],
            'features': [
                {'name': 'X', 'type': 'continuous', 'range': '[0, 1]'},
                {'name': 'Y', 'type': 'continuous', 'range': '[0, 1]'},
            ],
            'objectives': [{'name': 'Yield', 'type': 'maximize'}],
            'tweaks': {},
            'x_feat': 'X',
            'y_feat': 'Y',
            'color_obj': 'Yield',
        }

        with patch.object(routes, 'run_bo_surface', side_effect=cancellable_surface):
            with app.test_client() as client:
                response = client.post('/bo_surface_submit', json=payload)
                self.assertEqual(response.status_code, 202)
                job_id = response.get_json()['job_id']
                self.assertTrue(started.wait(1))
                self.assertEqual(client.post(f'/job_cancel/{job_id}').status_code, 200)

                for _ in range(100):
                    status = client.get(f'/job_status/{job_id}').get_json()
                    if status['status'] == 'cancelled':
                        break
                    time.sleep(0.01)
                self.assertEqual(status['status'], 'cancelled')


if __name__ == '__main__':
    unittest.main()
