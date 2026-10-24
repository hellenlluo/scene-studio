import { Line } from "@react-three/drei";
import { useMemo } from "react";
import * as THREE from "three";

import type { BalanceCheck } from "../api/client";
import { type Theme, themeToken } from "../theme";

const CONTACT_RADIUS_M = 0.006;
const COM_RADIUS_M = 0.014;

type Point2 = [number, number];
type Point3 = [number, number, number];

interface EvidenceColours {
  contact: string;
  wrench: string;
  com: string;
  pass: string;
  fail: string;
}

// Debug geometry must never steal a click from the scene mesh beneath it.
function ignoreRaycast() {}

/** Project the backend's Z-up scene coordinates into three.js's Y-up world. */
function renderPoint([x, y, z]: Point3): [number, number, number] {
  return [x, z, -y];
}

function closestOnSegment(point: Point2, start: Point2, end: Point2): Point2 {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared === 0) return start;
  const t = Math.max(
    0,
    Math.min(
      1,
      ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / lengthSquared,
    ),
  );
  return [start[0] + t * dx, start[1] + t * dy];
}

function closestOnPolygon(point: Point2, polygon: Point2[]): Point2 | null {
  if (!polygon.length) return null;
  if (polygon.length === 1) return polygon[0];
  let closest = polygon[0];
  let best = Number.POSITIVE_INFINITY;
  for (let i = 0; i < polygon.length; i += 1) {
    const candidate = closestOnSegment(
      point,
      polygon[i],
      polygon[(i + 1) % polygon.length],
    );
    const distance =
      (candidate[0] - point[0]) ** 2 + (candidate[1] - point[1]) ** 2;
    if (distance < best) {
      closest = candidate;
      best = distance;
    }
  }
  return closest;
}

function polygonGeometry(polygon: Point2[], height: number) {
  if (polygon.length < 3) return null;
  const geometry = new THREE.BufferGeometry();
  geometry.setFromPoints(
    polygon.map(([x, y]) => new THREE.Vector3(...renderPoint([x, y, height]))),
  );
  const indices: number[] = [];
  for (let index = 1; index < polygon.length - 1; index += 1)
    indices.push(0, index, index + 1);
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function ObjectBalanceEvidence({
  check,
  colours,
}: {
  check: BalanceCheck;
  colours: EvidenceColours;
}) {
  const contacts = (check.contact_points_m ?? []) as Point3[];
  const wrenchContacts = check.wrench_contacts ?? [];
  const polygon = useMemo(
    () => (check.support_polygon_xy_m ?? []) as Point2[],
    [check.support_polygon_xy_m],
  );
  const centre = check.centre_of_mass_m as Point3 | null | undefined;
  const supportHeight = contacts.length
    ? contacts.reduce((sum, point) => sum + point[2], 0) / contacts.length
    : 0;
  const hasSupport = contacts.length > 0;
  const geometry = useMemo(
    () => polygonGeometry(polygon, supportHeight + 0.001),
    [polygon, supportHeight],
  );

  if (!centre) return null;
  const projection: Point3 = [centre[0], centre[1], supportHeight];
  const nearest = closestOnPolygon([centre[0], centre[1]], polygon);
  const verdictColour = check.passed ? colours.pass : colours.fail;

  return (
    <group name={`balance:${check.object_id}`}>
      {geometry && (
        <mesh geometry={geometry} renderOrder={20} raycast={ignoreRaycast}>
          <meshBasicMaterial
            color={verdictColour}
            transparent
            opacity={0.24}
            side={THREE.DoubleSide}
            depthWrite={false}
          />
        </mesh>
      )}
      {polygon.length >= 2 && (
        <Line
          points={[...polygon, polygon[0]].map(([x, y]) =>
            renderPoint([x, y, supportHeight + 0.003]),
          )}
          color={verdictColour}
          lineWidth={2}
          depthTest={false}
          renderOrder={22}
          raycast={ignoreRaycast}
        />
      )}
      {contacts.map((point, index) => (
        <mesh
          key={`${point.join(":")}:${index}`}
          position={renderPoint(point)}
          renderOrder={24}
          raycast={ignoreRaycast}
        >
          <sphereGeometry args={[CONTACT_RADIUS_M, 12, 8]} />
          <meshBasicMaterial color={colours.contact} depthTest={false} />
        </mesh>
      ))}
      {wrenchContacts.map((contact, index) => {
        const point = contact.position_m as Point3;
        const normal = contact.normal_m as Point3;
        const end: Point3 = [
          point[0] + normal[0] * 0.06,
          point[1] + normal[1] * 0.06,
          point[2] + normal[2] * 0.06,
        ];
        return (
          <group key={`wrench:${point.join(":")}:${index}`}>
            <mesh
              position={renderPoint(point)}
              renderOrder={25}
              raycast={ignoreRaycast}
            >
              <sphereGeometry args={[0.004, 10, 8]} />
              <meshBasicMaterial color={colours.wrench} depthTest={false} />
            </mesh>
            <Line
              points={[renderPoint(point), renderPoint(end)]}
              color={colours.wrench}
              lineWidth={2}
              depthTest={false}
              renderOrder={25}
              raycast={ignoreRaycast}
            />
          </group>
        );
      })}
      {hasSupport && (
        <Line
          points={[renderPoint(centre), renderPoint(projection)]}
          color={colours.com}
          lineWidth={2}
          dashed
          dashSize={0.035}
          gapSize={0.02}
          depthTest={false}
          renderOrder={25}
          raycast={ignoreRaycast}
        />
      )}
      <mesh
        position={renderPoint(centre)}
        renderOrder={26}
        raycast={ignoreRaycast}
      >
        <sphereGeometry args={[COM_RADIUS_M, 16, 12]} />
        <meshBasicMaterial color={colours.com} depthTest={false} />
      </mesh>
      {hasSupport && (
        <mesh
          position={renderPoint(projection)}
          rotation={[-Math.PI / 2, 0, 0]}
          renderOrder={26}
          raycast={ignoreRaycast}
        >
          <ringGeometry args={[0.025, 0.04, 20]} />
          <meshBasicMaterial
            color={colours.com}
            side={THREE.DoubleSide}
            depthTest={false}
          />
        </mesh>
      )}
      {nearest && (
        <Line
          points={[
            renderPoint(projection),
            renderPoint([nearest[0], nearest[1], supportHeight]),
          ]}
          color={verdictColour}
          lineWidth={3}
          depthTest={false}
          renderOrder={25}
          raycast={ignoreRaycast}
        />
      )}
    </group>
  );
}

export function BalanceOverlay({
  checks,
  theme,
}: {
  checks: BalanceCheck[];
  theme: Theme;
}) {
  const colours = useMemo<EvidenceColours>(
    () => ({
      contact: themeToken(
        "--evidence-contact",
        theme === "dark" ? "#c4a46b" : "#8f7040",
      ),
      wrench: themeToken(
        "--evidence-wrench",
        theme === "dark" ? "#79a5ae" : "#4c7b85",
      ),
      com: themeToken(
        "--evidence-com",
        theme === "dark" ? "#a693b1" : "#766583",
      ),
      pass: themeToken(
        "--evidence-pass",
        theme === "dark" ? "#779884" : "#527863",
      ),
      fail: themeToken(
        "--evidence-fail",
        theme === "dark" ? "#bb7a7c" : "#955d60",
      ),
    }),
    [theme],
  );

  return (
    <>
      {checks.map((check) => (
        <ObjectBalanceEvidence
          key={check.object_id}
          check={check}
          colours={colours}
        />
      ))}
    </>
  );
}
