import unittest

from app import app


class DeploymentMetadataTests(unittest.TestCase):
    def test_health_endpoint(self):
        with app.test_client() as client:
            response = client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['application'], 'CEDOS')

    def test_version_endpoint_has_traceability_fields(self):
        with app.test_client() as client:
            response = client.get('/version')
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['application'], 'CEDOS')
        self.assertIn('application_commit', payload)
        self.assertIn('edos_foundation_commit', payload)


if __name__ == '__main__':
    unittest.main()
