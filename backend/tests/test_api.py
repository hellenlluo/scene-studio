import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "grey").save(buf, format="PNG")
    return buf.getvalue()


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_rejects_non_image_upload():
    response = client.post(
        "/api/jobs",
        files={"image": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_upload_creates_pending_job():
    response = client.post(
        "/api/jobs",
        files={"image": ("room.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["state"] == "pending"
    assert body["scene_id"] is None

    # The pipeline stages are stubs, so the background run is expected to fail —
    # what matters here is that the failure lands in the job record rather than
    # taking down the request.
    status = client.get(f"/api/jobs/{body['id']}").json()
    assert status["state"] in {"pending", "running", "failed"}


def test_unknown_job_404s():
    assert client.get("/api/jobs/does-not-exist").status_code == 404
