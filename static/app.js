/* Shared client-side helpers (intentionally tiny - per-page logic lives
   in each .html for clarity). */

window.api = {
  predict: async (text) => {
    const r = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text}),
    });
    if (!r.ok) throw new Error("predict failed: " + r.status);
    return r.json();
  },
  stats: async () => {
    const r = await fetch("/api/stats");
    if (!r.ok) throw new Error("stats failed: " + r.status);
    return r.json();
  },
};
