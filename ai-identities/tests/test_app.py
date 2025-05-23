import unittest
import requests
import json
import time

class TestMistralAPI(unittest.TestCase):
    BASE_URL = "http://localhost:5001"  # Update with your server URL
    TEST_API_KEY = "nPhMGUsSCuNqj65iX8z8JojNnCMnBLjy"  # Replace with valid test key
    TEST_MODEL = "open-mistral-nemo"  # Or any valid Mistral model
    PROVIDER = "mistral"

    def test_test_connection(self):
        """Test the /api/test-connection endpoint with Mistral provider"""
        url = f"{self.BASE_URL}/api/test-connection"
        headers = {"Content-Type": "application/json"}
        payload = {
            "api_key": self.TEST_API_KEY,
            "provider": self.PROVIDER,
            "model": self.TEST_MODEL,
            "temperature": 0.7
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        data = response.json()

        print(f"Test Connection Response: {json.dumps(data, indent=2)}")

        self.assertEqual(response.status_code, 200)
        self.assertIn("status", data)
        self.assertTrue(data["status"] in ["success", "error"])

        if data["status"] == "success":
            self.assertIn("response_preview", data)
            self.assertTrue(isinstance(data["response_preview"], str))
        else:
            self.assertIn("message", data)
            print(f"Test Connection Failed: {data['message']}")

    def test_identify_model(self):
        """Test the /api/identify-model endpoint with Mistral provider"""
        url = f"{self.BASE_URL}/api/identify-model"
        headers = {"Content-Type": "application/json"}
        payload = {
            "api_key": self.TEST_API_KEY,
            "provider": self.PROVIDER,
            "model": self.TEST_MODEL,
            "temperature": 0.7,
            "num_samples": 10,  # Reduced for testing
            "batch_size": 2
        }

        start_time = time.time()
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        duration = time.time() - start_time
        data = response.json()

        print(f"Identify Model Response ({duration:.2f}s): {json.dumps(data, indent=2)}")

        self.assertEqual(response.status_code, 200)
        self.assertIn("status", data)
        self.assertTrue(data["status"].startswith("success"))
        self.assertIn("predicted_model", data)
        self.assertIn("confidence", data)
        self.assertIn("top_predictions", data)
        self.assertIsInstance(data["top_predictions"], list)

        if data["status"] == "success":
            self.assertGreater(float(data["confidence_value"]), 0)
        elif data["status"] == "success_no_overlap":
            print("Warning: No word overlap with training set")
        elif data["status"] == "success_unrecognized":
            print("Warning: Model was not recognized")

    @classmethod
    def setUpClass(cls):
        """Optional: Verify the API key and model before running tests"""
        # You could add a quick connection test here to fail fast if credentials are invalid
        pass

if __name__ == '__main__':
    unittest.main()
