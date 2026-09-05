"""The scenes API, exercised through the seed path the frontend will use.

These run against seeded fixture scenes rather than mocks, so they cover the same
route the browser takes: seed -> list -> fetch -> repair -> fetch again.
"""

import re

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


# --- the physics bundle --------------------------------------------------------


def test_physics_bundle_carries_the_mjcf_and_its_assets():
    """One request, so the client never has to reimplement `mjcf`'s asset naming."""
    response = client.get(f"/api/scenes/{KITCHEN}/physics")
    assert response.status_code == 200
    body = response.json()

    assert body["mjcf"].startswith("<mujoco")
    assert "<worldbody" in body["mjcf"]
    # Every mesh the MJCF names has somewhere to be fetched from.
    for name in re.findall(r'file="([^"]*)"', body["mjcf"]):
        assert name in body["meshes"], f"{name} is named but not served"


def test_physics_bundle_urls_are_storage_relative():
    """The browser prepends `/storage/`; an absolute filesystem path is unusable to
    it and breaks the moment the storage directory moves."""
    body = client.get(f"/api/scenes/{KITCHEN}/physics").json()
    for url in body["meshes"].values():
        assert not url.startswith("/")
        assert ".." not in url


def test_physics_bundle_follows_a_repair():
    """Built from the stored graph, not read off disk. `scene.xml` is rewritten in
    place on every repair, so a client holding a stale URL would compile the
    pre-repair scene."""
    client.post(f"/api/scenes/{SUNK}/repair")
    body = client.get(f"/api/scenes/{SUNK}/physics").json()

    assert body["mjcf"].startswith("<mujoco")
    for name in re.findall(r'file="([^"]*)"', body["mjcf"]):
        assert name in body["meshes"]


def test_physics_bundle_404s_for_an_unknown_scene():
    assert client.get("/api/scenes/nope/physics").status_code == 404


# --- committing an edit --------------------------------------------------------


def _mug_position(scene_id):
    spec = client.get(f"/api/scenes/{scene_id}").json()["spec"]
    return next(o for o in spec["graph"]["objects"] if o["object_id"] == "mug")["position_m"]


def test_an_edit_moves_the_object_and_persists():
    before = _mug_position(KITCHEN)
    target = [before[0] + 0.12, before[1], before[2]]

    response = client.put(
        f"/api/scenes/{KITCHEN}",
        json={"objects": [{"object_id": "mug", "position_m": target}], "resolve": False},
    )
    assert response.status_code == 200
    # Read it back from the server rather than trusting the response body.
    assert _mug_position(KITCHEN)[0] == pytest.approx(target[0], abs=1e-6)


def test_an_edited_value_is_marked_as_the_user_s():
    """`Provenance.USER` records that a person chose this, as distinct from a model
    predicting it — worth keeping whether or not the solver ever holds it fixed."""
    client.put(
        f"/api/scenes/{KITCHEN}",
        json={"objects": [{"object_id": "mug", "scale": 1.1}], "resolve": False},
    )
    spec = client.get(f"/api/scenes/{KITCHEN}").json()["spec"]
    mug = next(o for o in spec["graph"]["objects"] if o["object_id"] == "mug")

    assert mug["provenance"]["scale"] == "user"


def test_committing_re_certifies():
    """The certificate returned is the honest answer to "what did my edit do",
    including when the answer is that it made things worse."""
    body = client.put(
        f"/api/scenes/{KITCHEN}",
        json={
            # Half a metre up: floating, so the scale axis has to notice.
            "objects": [{"object_id": "mug", "position_m": [0.0, 0.0, 1.4]}],
            "resolve": False,
        },
    ).json()

    mug = next(c for c in body["spec"]["certificate"]["scale"] if c["object_id"] == "mug")
    assert not mug["passed"], "a mug left in mid-air should fail its support check"


def test_committing_does_not_repair():
    """Repair is a separate, deliberate action. An edit that silently moved objects
    the user did not touch would be a surprising thing for a drag to do."""
    body = client.put(
        f"/api/scenes/{KITCHEN}",
        json={
            "objects": [{"object_id": "mug", "position_m": [0.0, 0.0, 1.4]}],
            "resolve": False,
        },
    ).json()

    assert body["spec"]["repairs_applied"] == []


def test_the_exports_are_rewritten_so_the_viewer_sees_the_edit():
    body = client.put(
        f"/api/scenes/{KITCHEN}",
        json={"objects": [{"object_id": "mug", "scale": 1.2}], "resolve": False},
    ).json()

    for path in (body["spec"]["exports"]["gltf_path"], body["spec"]["exports"]["mjcf_path"]):
        assert client.get(f"/storage/{path}").status_code == 200


def test_an_unknown_object_is_rejected():
    for payload in (
        {"objects": [{"object_id": "ghost", "scale": 1.1}]},
        {"anchors": [{"object_id": "ghost", "axis": 0, "value_m": 1.0}]},
        {"objects": [{"object_id": "mug", "supported_by": "ghost"}]},
    ):
        assert client.put(f"/api/scenes/{KITCHEN}", json=payload).status_code == 422


def test_editing_an_unknown_scene_404s():
    assert client.put("/api/scenes/nope", json={"objects": []}).status_code == 404


def test_a_re_solve_keeps_an_edit_it_has_no_reason_to_undo():
    """Warm start, not a constraint — but nothing opposes a sideways nudge, so it
    survives. This is what makes committing feel like the edit took."""
    before = _mug_position(KITCHEN)
    target = [before[0] + 0.05, before[1], before[2]]
    client.put(
        f"/api/scenes/{KITCHEN}",
        json={"objects": [{"object_id": "mug", "position_m": target}], "resolve": True},
    )

    assert _mug_position(KITCHEN)[0] == pytest.approx(target[0], abs=0.02)


def test_a_re_solve_pulls_an_impossible_edit_back():
    """The other half, and the reason a drag is seeded rather than held: put the mug
    in mid-air and the contact term brings it back down onto the table.

    An edit held as a hard constraint would leave it floating and fail the scene on
    the user's own instruction. Seeding lets the measurement argue back."""
    client.put(
        f"/api/scenes/{KITCHEN}",
        json={
            "objects": [{"object_id": "mug", "position_m": [0.0, 0.0, 1.4]}],
            "resolve": True,
        },
    )
    spec = client.get(f"/api/scenes/{KITCHEN}").json()["spec"]
    mug = next(c for c in spec["certificate"]["scale"] if c["object_id"] == "mug")

    assert _mug_position(KITCHEN)[2] < 1.2, "the solver brought it back toward its support"
    assert mug["passed"], "and it is resting again"


def test_a_re_solve_leaves_the_value_derived_not_the_user_s():
    """A consequence of seeding rather than pinning, asserted so it is not a surprise.

    `Provenance.USER` is written when the edit is applied and then overwritten by
    the solve, because with a warm start the number that ends up stored is the
    solver's refinement of the user's drag, not the drag itself. Claiming the user
    set that exact value would be false. The marking only survives `resolve: false`.

    If edits ever need to be held rather than seeded, this is the assertion that
    will have to change, and it should change deliberately.
    """
    client.put(
        f"/api/scenes/{KITCHEN}",
        json={"objects": [{"object_id": "mug", "position_m": [-0.05, 0.0, 0.8]}], "resolve": True},
    )
    spec = client.get(f"/api/scenes/{KITCHEN}").json()["spec"]
    mug = next(o for o in spec["graph"]["objects"] if o["object_id"] == "mug")

    assert mug["provenance"]["position_m"] == "derived"
