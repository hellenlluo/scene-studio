import { describe, expect, it } from "vitest";

import { supportChangeMessage } from "./supportChanges";

describe("supportChangeMessage", () => {
  const names = new Map([
    ["mug", "mug"],
    ["desk", "desk"],
    ["side_table", "side table"],
  ]);

  it("names the floor when an object is moved off furniture", () => {
    expect(
      supportChangeMessage(
        {
          object_id: "mug",
          previous_support_id: "desk",
          new_support_id: null,
        },
        names,
      ),
    ).toBe("mug support changed: desk → floor");
  });

  it("names both objects when an object is reparented", () => {
    expect(
      supportChangeMessage(
        {
          object_id: "mug",
          previous_support_id: "desk",
          new_support_id: "side_table",
        },
        names,
      ),
    ).toBe("mug support changed: desk → side table");
  });

  it("falls back to ids when display labels are unavailable", () => {
    expect(
      supportChangeMessage(
        {
          object_id: "unknown_child",
          previous_support_id: null,
          new_support_id: "unknown_parent",
        },
        names,
      ),
    ).toBe("unknown_child support changed: floor → unknown_parent");
  });
});
