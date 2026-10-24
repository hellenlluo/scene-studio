"""General 3-D contact-wrench feasibility at the photographed pose.

The support-polygon test in :mod:`app.certify.balance` is exact for an object on
horizontal, frictionless support. It is not a general static-equilibrium test: a
book leaning against a basket wall can have its centre of mass outside the narrow
bottom patch and still be held by the wall normal and friction.

This module reads contacts from the same MJCF MuJoCo later settles. For each object
and contact, it builds a conservative polyhedral approximation of MuJoCo's
five-dimensional friction cone (two sliding, torsional, and two rolling terms).
Static equilibrium is then a small linear feasibility problem over non-negative
cone-generator weights.
"""

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import linprog

from app.export import mjcf
from app.schemas import SceneGraph, SceneObject

__all__ = ["ContactWrench", "contacts_by_object", "is_feasible", "is_non_horizontal"]


@dataclass(frozen=True)
class ContactWrench:
    """One MuJoCo contact expressed as forces acting on the inspected object."""

    position_m: tuple[float, float, float]
    normal_m: tuple[float, float, float]
    tangent_1_m: tuple[float, float, float]
    tangent_2_m: tuple[float, float, float]
    # sliding-1, sliding-2, torsional, rolling-1, rolling-2
    friction: tuple[float, float, float, float, float]
    dimensions: int
    other_object_id: str | None
    distance_m: float


def _body_owners(model, graph: SceneGraph) -> dict[int, str]:
    owners: dict[int, str] = {}
    for obj in graph.objects:
        for part in obj.parts:
            try:
                body_id = int(model.body(mjcf.body_name(obj.object_id, part.part_id)).id)
            except KeyError:
                continue
            owners[body_id] = obj.object_id
    return owners


def contacts_by_object(
    graph: SceneGraph,
    contact_tolerance_m: float = 0.0,
) -> dict[str, list[ContactWrench]]:
    """Extract initial 3-D contacts from the exact model used for certification.

    Contacts separated by less than ``contact_tolerance_m`` are evidence too. The
    rest of certification already treats such a small positive support gap as
    contact: reconstructed meshes rarely land exactly flush, and gravity closes
    the gap on the first simulation step. Without the same tolerance here, a book
    0.1 mm above a basket can alternate between having a wall wrench and having no
    basket contact at all depending on the exact frame the user commits.

    MuJoCo generates a contact when the pair's combined geom margins overlap, so
    half of the requested tolerance is assigned to every geom. This changes only
    the temporary evidence model; the exported/live simulation keeps zero margins.

    ``Contact.frame`` points from geom 1 toward geom 2. The compressive normal
    acting on geom 1 is therefore its negative, while geom 2 receives the positive
    direction. Tangent signs need no corresponding flip because both signs are
    included in the friction cone.
    """

    model = mjcf.load_model(graph)
    if contact_tolerance_m > 0.0:
        model.geom_margin[:] = contact_tolerance_m / 2.0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    owners = _body_owners(model, graph)
    contacts: dict[str, list[ContactWrench]] = {obj.object_id: [] for obj in graph.objects}

    for index in range(data.ncon):
        contact = data.contact[index]
        body_1 = int(model.geom_bodyid[contact.geom1])
        body_2 = int(model.geom_bodyid[contact.geom2])
        object_1 = owners.get(body_1)
        object_2 = owners.get(body_2)
        # Contacts between convex parts of one rigid object are internal and cannot
        # support that object. World/world cannot occur; the same branch covers it.
        if object_1 == object_2:
            continue

        frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
        friction = tuple(float(value) for value in contact.friction[:5])
        position = tuple(float(value) for value in contact.pos)

        if object_1 is not None:
            contacts[object_1].append(
                ContactWrench(
                    position_m=position,
                    normal_m=tuple(float(value) for value in -frame[0]),
                    tangent_1_m=tuple(float(value) for value in frame[1]),
                    tangent_2_m=tuple(float(value) for value in frame[2]),
                    friction=friction,
                    dimensions=int(contact.dim),
                    other_object_id=object_2,
                    distance_m=float(contact.dist),
                )
            )
        if object_2 is not None:
            contacts[object_2].append(
                ContactWrench(
                    position_m=position,
                    normal_m=tuple(float(value) for value in frame[0]),
                    tangent_1_m=tuple(float(value) for value in frame[1]),
                    tangent_2_m=tuple(float(value) for value in frame[2]),
                    friction=friction,
                    dimensions=int(contact.dim),
                    other_object_id=object_1,
                    distance_m=float(contact.dist),
                )
            )

    return contacts


def is_non_horizontal(contact: ContactWrench, max_tilt_deg: float = 30.0) -> bool:
    """Whether a contact supplies a materially sideways normal force."""

    normal = np.asarray(contact.normal_m, dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        return False
    return abs(float(normal[2]) / norm) < np.cos(np.deg2rad(max_tilt_deg))


def _generators(contact: ContactWrench, centre_m: np.ndarray) -> list[np.ndarray]:
    """Polyhedral wrench generators for one MuJoCo friction cone.

    Every generator carries one unit of compressive normal force. Its paired
    tangential force or contact moment is bounded by the corresponding MuJoCo
    coefficient. Convex combinations produce a conservative diamond approximation.
    """

    point = np.asarray(contact.position_m, dtype=float)
    normal = np.asarray(contact.normal_m, dtype=float)
    tangent_1 = np.asarray(contact.tangent_1_m, dtype=float)
    tangent_2 = np.asarray(contact.tangent_2_m, dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        return []
    normal /= norm
    offset = point - centre_m
    slide_1, slide_2, torsion, roll_1, roll_2 = contact.friction

    perturbations: list[tuple[np.ndarray, np.ndarray]] = [
        (slide_1 * tangent_1, np.zeros(3)),
        (-slide_1 * tangent_1, np.zeros(3)),
        (slide_2 * tangent_2, np.zeros(3)),
        (-slide_2 * tangent_2, np.zeros(3)),
    ]
    if contact.dimensions >= 4:
        perturbations.extend(
            [
                (np.zeros(3), torsion * normal),
                (np.zeros(3), -torsion * normal),
            ]
        )
    if contact.dimensions >= 6:
        perturbations.extend(
            [
                (np.zeros(3), roll_1 * tangent_1),
                (np.zeros(3), -roll_1 * tangent_1),
                (np.zeros(3), roll_2 * tangent_2),
                (np.zeros(3), -roll_2 * tangent_2),
            ]
        )
    generators: list[np.ndarray] = []
    for force_offset, moment_offset in perturbations:
        force = normal + force_offset
        moment = np.cross(offset, force) + moment_offset
        generators.append(np.concatenate((force, moment)))
    return generators


def is_feasible(obj: SceneObject, contacts: list[ContactWrench], centre_m: np.ndarray) -> bool:
    """Whether these contacts can hold ``obj`` in static equilibrium under gravity."""

    del obj  # retained in the boundary for future articulated/per-part constraints
    generators = [
        generator
        for contact in contacts
        for generator in _generators(contact, np.asarray(centre_m, dtype=float))
    ]
    if not generators:
        return False

    matrix = np.column_stack(generators)
    # Contact forces balance unit downward gravity. Gravity acts at the COM, so it
    # contributes no torque in this frame. Normalisation removes unreliable mass
    # magnitude without changing per-object feasibility.
    target = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    result = linprog(
        c=np.zeros(matrix.shape[1]),
        A_eq=matrix,
        b_eq=target,
        bounds=(0.0, None),
        method="highs",
        options={
            "dual_feasibility_tolerance": 1e-8,
            "primal_feasibility_tolerance": 1e-8,
        },
    )
    return bool(result.success)
