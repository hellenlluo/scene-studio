"""The scenes API, exercised end to end against stored scenes.

These run against hand-authored scenes inserted into the database rather than
mocks, so they cover the same route the browser takes: list -> fetch -> repair ->
fetch again. The scenes come from `tests.fixtures.seeded` because a real
reconstruction costs minutes and paid API calls per test.
"""

import re

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.fixtures.seeded import SOUND, SUNK, seed

client = TestClient(app)


@pytest.fixture(autouse=True)
def seeded():
    """Re-seed before each test. Repair mutates rows, so tests must not share state."""
    seed()


# --- seeding ------------------------------------------------------------------


def test_the_fixtures_give_one_passing_and_one_failing_scene():
    """A viewer only ever tested against a passing scene tells you nothing about
    whether it renders failure correctly."""
    scenes = {s["id"]: s for s in client.get("/api/scenes").json()}
    assert scenes[SOUND]["certified"] is True
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
    body = client.get(f"/api/scenes/{SOUND}").json()
    assert body["spec"]["scene_id"] == SOUND
    assert body["updated_at"]


def test_export_paths_are_storage_relative():
    """Absolute filesystem paths are unusable to a browser; the client builds the
    URL as /storage/{path}."""
    exports = client.get(f"/api/scenes/{SOUND}").json()["spec"]["exports"]
    for path in (exports["gltf_path"], exports["mjcf_path"]):
        assert not path.startswith("/")
        assert path.startswith("scenes/")


def test_the_exported_files_are_actually_served():
    exports = client.get(f"/api/scenes/{SOUND}").json()["spec"]["exports"]
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
    body = client.post(f"/api/scenes/{SOUND}/repair").json()
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
    response = client.get(f"/api/scenes/{SOUND}/physics")
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
    body = client.get(f"/api/scenes/{SOUND}/physics").json()
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
    before = _mug_position(SOUND)
    target = [before[0] + 0.12, before[1], before[2]]

    response = client.put(
        f"/api/scenes/{SOUND}",
        json={"objects": [{"object_id": "mug", "position_m": target}], "resolve": False},
    )
    assert response.status_code == 200
    # Read it back from the server rather than trusting the response body.
    assert _mug_position(SOUND)[0] == pytest.approx(target[0], abs=1e-6)


def test_an_edited_value_is_marked_as_the_user_s():
    """`Provenance.USER` records that a person chose this, as distinct from a model
    predicting it — worth keeping whether or not the solver ever holds it fixed."""
    client.put(
        f"/api/scenes/{SOUND}",
        json={"objects": [{"object_id": "mug", "scale": 1.1}], "resolve": False},
    )
    spec = client.get(f"/api/scenes/{SOUND}").json()["spec"]
    mug = next(o for o in spec["graph"]["objects"] if o["object_id"] == "mug")

    assert mug["provenance"]["scale"] == "user"


def test_committing_re_certifies():
    """The certificate returned is the honest answer to "what did my edit do",
    including when the answer is that it made things worse."""
    body = client.put(
        f"/api/scenes/{SOUND}",
        json={
            # Half a metre up: floating, so the scale axis has to notice.
            "objects": [{"object_id": "mug", "position_m": [0.0, 0.0, 1.4]}],
            "resolve": False,
        },
    ).json()

    spec = body["scene"]["spec"]
    mug = next(c for c in spec["certificate"]["scale"] if c["object_id"] == "mug")
    assert not mug["passed"], "a mug left in mid-air should fail its support check"


def test_committing_without_resolving_does_not_repair():
    """`resolve: false` stores the edit and nothing else.

    Repair rides on the same flag as the solve, because both are corrections and a
    client asking for the pose to be stored verbatim wants neither. The flag is the
    whole gate: with it off there is no free set, so there is nothing repair would
    be allowed to touch even if it ran.
    """
    body = client.put(
        f"/api/scenes/{SOUND}",
        json={
            "objects": [{"object_id": "mug", "position_m": [0.0, 0.0, 1.4]}],
            "resolve": False,
        },
    ).json()

    assert body["scene"]["spec"]["repairs_applied"] == []
    assert body["actions"] == []


def test_committing_repairs_only_what_the_edit_reached():
    """A commit finishes the job it started, without rearranging the rest of the room.

    Repair used to be refused here outright, which kept the scene honest and left
    the user with a visibly broken result and a button to find. Scoping it to the
    edited objects and their dependents answers the original objection — nothing
    outside the set is ever proposed — while letting the commit fix its own mess.
    """
    body = client.put(
        f"/api/scenes/{SOUND}",
        json={
            # Left floating: something for repair to find, on the object touched.
            "objects": [{"object_id": "mug", "position_m": [0.0, 0.0, 1.4]}],
            "resolve": True,
        },
    ).json()

    touched = {action["target_id"] for action in body["actions"]}
    assert touched <= {"mug"}, f"repair reached objects the edit did not: {touched - {'mug'}}"
    # Reported, not silent — including proposals that were tried and reverted.
    assert body["scene"]["spec"]["repairs_applied"] == body["actions"]


def test_the_exports_are_rewritten_so_the_viewer_sees_the_edit():
    body = client.put(
        f"/api/scenes/{SOUND}",
        json={"objects": [{"object_id": "mug", "scale": 1.2}], "resolve": False},
    ).json()

    exports = body["scene"]["spec"]["exports"]
    for path in (exports["gltf_path"], exports["mjcf_path"]):
        assert client.get(f"/storage/{path}").status_code == 200


def test_an_unknown_object_is_rejected():
    for payload in (
        {"objects": [{"object_id": "ghost", "scale": 1.1}]},
        {"anchors": [{"object_id": "ghost", "axis": 0, "value_m": 1.0}]},
        {"objects": [{"object_id": "mug", "supported_by": "ghost"}]},
    ):
        assert client.put(f"/api/scenes/{SOUND}", json=payload).status_code == 422


def test_editing_an_unknown_scene_404s():
    assert client.put("/api/scenes/nope", json={"objects": []}).status_code == 404


def test_a_re_solve_keeps_an_edit_it_has_no_reason_to_undo():
    """Warm start, not a constraint — but nothing opposes a sideways nudge, so it
    survives. This is what makes committing feel like the edit took."""
    before = _mug_position(SOUND)
    target = [before[0] + 0.05, before[1], before[2]]
    client.put(
        f"/api/scenes/{SOUND}",
        json={"objects": [{"object_id": "mug", "position_m": target}], "resolve": True},
    )

    assert _mug_position(SOUND)[0] == pytest.approx(target[0], abs=0.02)


def test_a_long_drag_into_clear_space_survives_the_solve():
    """The reported bug: a drag with nothing opposing it used to be erased entirely.

    Measured on `room2` before the fix, a 0.5 m drag and a 1.0 m drag of the lamp
    each kept **0%** — the stale depth centre is three of the six depth residuals
    and simply outvoted the seed. `solve._edit_priors` re-aims those three at what
    the user set, so a move into empty space now costs nothing to keep.

    The cabinet, because it stands on the floor: a drag along the floor is opposed
    by nothing, which is what isolates the depth term as the thing under test.
    """

    def cabinet_x():
        spec = client.get(f"/api/scenes/{SOUND}").json()["spec"]
        return next(o for o in spec["graph"]["objects"] if o["object_id"] == "cabinet")[
            "position_m"
        ][0]

    before = cabinet_x()
    target = before + 0.6
    client.put(
        f"/api/scenes/{SOUND}",
        json={
            "objects": [{"object_id": "cabinet", "position_m": [target, 0.0, 0.46]}],
            "resolve": True,
        },
    )

    kept = (cabinet_x() - before) / 0.6
    assert kept > 0.9, f"the solve kept only {100 * kept:.0f}% of a 0.6 m drag"


def test_a_re_solve_pulls_an_impossible_edit_back():
    """The other half, and the reason a drag is seeded rather than held: put the mug
    in mid-air and the contact term brings it back down onto the table.

    An edit held as a hard constraint would leave it floating and fail the scene on
    the user's own instruction. Seeding lets the measurement argue back."""
    client.put(
        f"/api/scenes/{SOUND}",
        json={
            "objects": [{"object_id": "mug", "position_m": [0.0, 0.0, 1.4]}],
            "resolve": True,
        },
    )
    spec = client.get(f"/api/scenes/{SOUND}").json()["spec"]
    mug = next(c for c in spec["certificate"]["scale"] if c["object_id"] == "mug")

    assert _mug_position(SOUND)[2] < 1.2, "the solver brought it back toward its support"
    assert mug["passed"], "and it is resting again"


def test_a_re_solve_keeps_the_user_marking_on_what_the_user_set():
    """`Provenance.USER` survives the solve, and has to.

    This assertion used to read `derived`, on the reasoning that the stored number
    is the solver's refinement of the drag rather than the drag itself. That was
    defensible while nothing read the flag. It is not now: `solve._edit_priors`
    reads it to decide whether the depth term aims at the measurement or at the
    user's placement, so stamping over it made the solve erase its own input — the
    first commit would honour a drag and the second would silently go back to being
    outvoted by the stale depth centre.

    What is stored is still the refinement, and `position_m` may differ from what
    was sent. The flag records *who decided this property*, which is the user, and
    that stays true however far physics then nudges it.
    """
    client.put(
        f"/api/scenes/{SOUND}",
        json={"objects": [{"object_id": "mug", "position_m": [-0.05, 0.0, 0.8]}], "resolve": True},
    )
    spec = client.get(f"/api/scenes/{SOUND}").json()["spec"]
    mug = next(o for o in spec["graph"]["objects"] if o["object_id"] == "mug")

    assert mug["provenance"]["position_m"] == "user"
    # Untouched properties are still the solver's, so this is not a blanket
    # "everything the user submitted is now theirs".
    assert mug["provenance"]["scale"] == "derived"


def test_a_second_commit_honours_a_drag_as_much_as_the_first():
    """The regression the provenance fix exists to prevent.

    With the marking erased by round one, round two re-read the stale depth centre
    and pulled the object home again — so a user who nudged something twice saw the
    second nudge behave completely differently from the first.
    """
    kept = []
    for _ in range(2):
        before = _mug_position(SOUND)
        target = [before[0] + 0.05, before[1], before[2]]
        client.put(
            f"/api/scenes/{SOUND}",
            json={"objects": [{"object_id": "mug", "position_m": target}], "resolve": True},
        )
        kept.append(_mug_position(SOUND)[0] - before[0])

    assert kept[1] == pytest.approx(kept[0], abs=0.01), (
        f"the second drag kept {1000 * kept[1]:.0f} mm where the first kept {1000 * kept[0]:.0f} mm"
    )


def test_dragging_an_object_onto_something_reparents_it():
    """The reported bug, end to end.

    The gizmo sends a position and nothing else — there is no parent in a drag —
    so `supported_by` used to survive the edit unchanged and stage 6 closed the gap
    to whatever the object *used* to rest on. Measured on `room2`, a book recorded
    as floor-supported and dropped on the side table came back from the commit at
    z=0.015: on the floor, underneath the table it had been dropped on.

    The cabinet stands on the floor here, so lifting it onto the table is a change
    of parent and not merely of height.
    """
    spec = client.get(f"/api/scenes/{SOUND}").json()["spec"]
    cabinet = next(o for o in spec["graph"]["objects"] if o["object_id"] == "cabinet")
    assert cabinet["supported_by"] is None, "the fixture should start on the floor"

    # Base at the tabletop (z=0.75), centred over it: dropped squarely on the table.
    client.put(
        f"/api/scenes/{SOUND}",
        json={
            "objects": [{"object_id": "cabinet", "position_m": [0.0, 0.0, 0.75 + 0.46]}],
            "resolve": True,
        },
    )

    spec = client.get(f"/api/scenes/{SOUND}").json()["spec"]
    cabinet = next(o for o in spec["graph"]["objects"] if o["object_id"] == "cabinet")
    assert cabinet["supported_by"] == "table", "the drop should have adopted the table"
    assert cabinet["position_m"][2] > 0.75, "and it should not have sunk back to the floor"


def test_an_explicit_parent_is_not_re_derived():
    """Re-deriving is for a drag, which carries no parent. When the user names one
    it is a decision about the present, not a memory of the past, and it stands."""
    client.put(
        f"/api/scenes/{SOUND}",
        json={
            # On the floor by geometry, declared on the table by the user.
            "objects": [
                {"object_id": "cabinet", "position_m": [1.2, 0.0, 0.46], "supported_by": "table"}
            ],
            "resolve": True,
        },
    )
    spec = client.get(f"/api/scenes/{SOUND}").json()["spec"]
    cabinet = next(o for o in spec["graph"]["objects"] if o["object_id"] == "cabinet")

    assert cabinet["supported_by"] == "table"
