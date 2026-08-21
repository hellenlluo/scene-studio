"""The scenes API, exercised through the seed path the frontend will use.

These run against seeded fixture scenes rather than mocks, so they cover the same
route the browser takes: seed -> list -> fetch -> repair -> fetch again.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.seed import seed

client = TestClient(app)

KITCHEN = "fixture-kitchen"
SUNK = "fixture-mug-sunk"


@pytest.fixture(autouse=True)
def seeded():
    """Re-seed before each test. Repair mutates rows, so tests must not share state."""
    seed()


# --- seeding ------------------------------------------------------------------


def test_seeding_gives_one_passing_and_one_failing_scene():
    """A viewer only ever tested against a passing scene tells you nothing about
    whether it renders failure correctly."""
    scenes = {s["id"]: s for s in client.get("/api/scenes").json()}
    assert scenes[KITCHEN]["certified"] is True
    assert scenes[SUNK]["certified"] is False


def test_seeding_is_idempotent():
    before = len(client.get("/api/scenes").json())
    seed()
    assert len(client.get("/api/scenes").json()) == before


def test_summaries_carry_the_axis_statuses():
    """The list view colours each scene without having to fetch every spec."""
    scenes = {s["id"]: s for s in client.get("/api/scenes").json()}
    assert scenes[SUNK]["axes"] == {
        "scale": "fail",
        "stability": "fail",
        "inertial": "pass",
        "cost": "pass",
    }


# --- fetching -----------------------------------------------------------------


def test_a_scene_comes_back_with_its_row_metadata():
    body = client.get(f"/api/scenes/{KITCHEN}").json()
    assert body["spec"]["scene_id"] == KITCHEN
    assert body["updated_at"]


def test_export_paths_are_storage_relative():
    """Absolute filesystem paths are unusable to a browser; the client builds the
    URL as /storage/{path}."""
    exports = client.get(f"/api/scenes/{KITCHEN}").json()["spec"]["exports"]
    for path in (exports["gltf_path"], exports["mjcf_path"]):
        assert not path.startswith("/")
        assert path.startswith("scenes/")


def test_the_exported_files_are_actually_served():
    exports = client.get(f"/api/scenes/{KITCHEN}").json()["spec"]["exports"]
    response = client.get(f"/storage/{exports['gltf_path']}")
    assert response.status_code == 200
    assert response.content[:4] == b"glTF"


def test_unknown_scene_404s():
    assert client.get("/api/scenes/does-not-exist").status_code == 404


# --- repair -------------------------------------------------------------------


def test_repairing_a_broken_scene_certifies_it():
    """The acceptance criterion for the whole viewer milestone, at the API level."""
    assert client.get(f"/api/scenes/{SUNK}").json()["spec"]["certificate"]["scale_status"] == "fail"

    body = client.post(f"/api/scenes/{SUNK}/repair").json()
    certificate = body["scene"]["spec"]["certificate"]

    assert certificate["scale_status"] == "pass"
    assert certificate["stability_status"] == "pass"
    assert body["converged"] is True
    assert [a["kind"] for a in body["actions"]] == ["snap_to_support"]


def test_the_repair_is_persisted():
    client.post(f"/api/scenes/{SUNK}/repair")
    certificate = client.get(f"/api/scenes/{SUNK}").json()["spec"]["certificate"]
    assert certificate["stability_status"] == "pass"


def test_repair_bumps_updated_at_so_the_glb_is_not_served_stale():
    """scene.glb is rewritten in place at the same URL. Without a version to hang
    off the request the browser serves its cached copy and the repair looks like it
    did nothing."""
    before = client.get(f"/api/scenes/{SUNK}").json()["updated_at"]
    after = client.post(f"/api/scenes/{SUNK}/repair").json()["scene"]["updated_at"]
    assert after != before


def test_repair_regenerates_the_exports():
    """A repaired scene whose MJCF still described the broken one would defeat the
    point of certifying at all."""
    exports = client.get(f"/api/scenes/{SUNK}").json()["spec"]["exports"]
    before = client.get(f"/storage/{exports['mjcf_path']}").text

    client.post(f"/api/scenes/{SUNK}/repair")
    after = client.get(f"/storage/{exports['mjcf_path']}").text

    assert before != after


def test_repairing_a_sound_scene_changes_nothing():
    body = client.post(f"/api/scenes/{KITCHEN}/repair").json()
    assert body["actions"] == []
    assert body["scene"]["spec"]["certificate"]["scale_status"] == "pass"


def test_repair_history_survives_a_second_repair():
    """Actions are appended rather than replaced, so a scene repaired twice still
    records what the first pass corrected. The second pass adds nothing here
    because there is nothing left to fix — which is the point: the history is not
    reset just because the latest run was a no-op."""
    client.post(f"/api/scenes/{SUNK}/repair")
    after_first = client.get(f"/api/scenes/{SUNK}").json()["spec"]["repairs_applied"]
    assert [a["kind"] for a in after_first] == ["snap_to_support"]

    client.post(f"/api/scenes/{SUNK}/repair")
    after_second = client.get(f"/api/scenes/{SUNK}").json()["spec"]["repairs_applied"]
    assert after_second == after_first


def test_repairing_an_unknown_scene_404s():
    assert client.post("/api/scenes/does-not-exist/repair").status_code == 404
