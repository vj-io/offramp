"use client";

import { THEME_KEY } from "@/lib/theme";

// Paper <-> blackboard. The label is chosen by CSS (see .theme-*), so no hydration mismatch.

export function ThemeToggle() {
  function toggle() {
    const root = document.documentElement;
    const dark =
      root.dataset.theme === "dark" ||
      (root.dataset.theme !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);
    root.dataset.theme = dark ? "light" : "dark";
    try {
      localStorage.setItem(THEME_KEY, root.dataset.theme);
    } catch {
      // storage blocked: the choice lasts for this page only
    }
  }
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label="Switch between Paper and Blackboard"
      className="sketch cursor-pointer bg-paper px-3 py-1 font-hand text-base text-ink hover:bg-note-plain"
    >
      <span className="theme-to-dark">Blackboard</span>
      <span className="theme-to-light">Paper</span>
    </button>
  );
}
