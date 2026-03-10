import requests
from pathlib import Path

# ==============================
# CONFIG
# ==============================

BASE_URL = "http://10.29.208.81:8004"
IMAGE_PATH = "test_images/vggface2-face/akash_face2_align.jpg"
MARK_ATTENDANCE_IMAGE_PATH = "test_images/vggface2-face/akash_face1_align.jpg"
TIMEOUT = 30




# ==============================
# API CLIENT
# ==============================

class APIClient:

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()

    def post(self, endpoint, data=None, files=None):

        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.post(
                url,
                data=data,
                files=files,
                timeout=TIMEOUT
            )

            print(f"\nPOST {endpoint}")
            print("Status:", response.status_code)

            try:
                print("Response:", response.json())
            except:
                print("Response:", response.text)

            return response

        except requests.exceptions.RequestException as e:
            print("Request failed:", e)
            return None


client = APIClient(BASE_URL)


# ==============================
# ROUTES
# ==============================

def register_face():

    if not Path(IMAGE_PATH).exists():
        print("Image not found:", IMAGE_PATH)
        return

    with open(IMAGE_PATH, "rb") as f:

        files = {
            "files": ("akash_face1_align.jpg", f, "image/jpeg")
        }

        data = {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "1234567890",
            "department": "Engineering",
            "role": "Employee",
            "threshold": 0.80
        }

        client.post("/faces/register", data=data, files=files)


def batch_attendance():

    if not Path(MARK_ATTENDANCE_IMAGE_PATH).exists():
        print("Image not found:", MARK_ATTENDANCE_IMAGE_PATH)
        return

    with open(MARK_ATTENDANCE_IMAGE_PATH, "rb") as f:

        files = [
            ("files", ("far_dis_face2_align.jpg", f, "image/jpeg"))
        ]

        data = {
            "camera_id": "CAM_01"
        }

        client.post("/attendance/batch", data=data, files=files)


# ==============================
# TEST RUNNER
# ==============================

def run_tests():

    print("\n===== API TEST START =====")

    # print("\n1️⃣ Register Face API")
    # register_face()

    print("\n2️⃣ Batch Attendance API")
    batch_attendance()

    print("\n===== API TEST COMPLETE =====")


# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":
    run_tests()