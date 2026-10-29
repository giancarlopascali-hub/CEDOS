import os
import unittest
from unittest.mock import patch

from app import app


class HostedUiTests(unittest.TestCase):
    def test_exploration_controls_start_disabled_for_default_ei(self):
        with app.test_client() as client:
            response = client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.count(b'exploration-control control-disabled'), 2)
        for control_id in (
            b'style-slider', b'style-number',
            b'tbo-exploration', b'tbo-exploration-number',
        ):
            self.assertRegex(response.data, rb'id="' + control_id + rb'"[^>]*disabled')

    def test_shutdown_button_is_visible_for_local_desktop(self):
        with patch.dict(os.environ, {}, clear=True), app.test_client() as client:
            response = client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="shutdown-btn"', response.data)

    def test_shutdown_button_and_endpoint_are_disabled_on_hugging_face(self):
        with patch.dict(os.environ, {'SPACE_ID': 'realgcp/CEDOS'}, clear=True), app.test_client() as client:
            page = client.get('/')
            shutdown = client.post('/shutdown')
        self.assertNotIn(b'id="shutdown-btn"', page.data)
        self.assertEqual(shutdown.status_code, 403)

    def test_hugging_face_hostname_also_hides_shutdown_button(self):
        with patch.dict(os.environ, {}, clear=True), app.test_client() as client:
            response = client.get('/', base_url='https://example.hf.space')
        self.assertNotIn(b'id="shutdown-btn"', response.data)

    def test_transbo_uses_per_prior_rules_without_an_overall_threshold(self):
        with app.test_client() as client:
            response = client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b'tbo-min-relevance-slider', response.data)
        self.assertNotIn(b'Minimum Combined Prior Relevance', response.data)
        self.assertIn(b'Maximum GP Training Rows', response.data)

    def test_advanced_bo_controls_exist_but_start_hidden(self):
        with app.test_client() as client:
            response = client.get('/')
        self.assertIn(b'id="bo-mo-method-item" class="tweak-item hidden"', response.data)
        self.assertIn(b'id="bo-large-data-options" class="tweak-item full-width hidden"', response.data)
        self.assertIn(b'Auto (recommended)', response.data)

    def test_hosted_profile_is_exposed_to_disable_unsafe_full_mode(self):
        with patch.dict(os.environ, {'SPACE_ID': 'realgcp/CEDOS'}, clear=True), app.test_client() as client:
            response = client.get('/')
        self.assertIn(b'data-hosted-environment="true"', response.data)
        self.assertIn(b'id="auto-access-token" type="password"', response.data)

    def test_local_auto_does_not_ask_for_a_hosted_token(self):
        with patch.dict(os.environ, {}, clear=True), app.test_client() as client:
            response = client.get('/')
        self.assertNotIn(b'id="auto-access-token"', response.data)


if __name__ == '__main__':
    unittest.main()
