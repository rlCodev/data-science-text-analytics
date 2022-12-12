import unittest
from starlette.testclient import TestClient

from api.main import app


class OpenApiTest(unittest.TestCase):
    """Test of the OpenAPI Swagger documentation page is reachable"""

    def setUp(self):
        self.app = TestClient(app)

    def test_open_api(self):
        expected = 200
        actual = self.app.get("/")
        self.assertEqual(actual.status_code, expected)


if __name__ == "__main__":
    unittest.main()