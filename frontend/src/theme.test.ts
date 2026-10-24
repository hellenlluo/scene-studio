import { describe, expect, it } from "vitest";

import { isTheme, resolveTheme } from "./theme";

describe("theme preference", () => {
  it("uses an explicit stored choice", () => {
    expect(resolveTheme("light")).toBe("light");
    expect(resolveTheme("dark")).toBe("dark");
  });

  it("defaults to dark when there is no explicit choice", () => {
    expect(resolveTheme(null)).toBe("dark");
  });

  it("ignores invalid stored values", () => {
    expect(isTheme("contrast")).toBe(false);
    expect(resolveTheme("contrast")).toBe("dark");
  });
});
