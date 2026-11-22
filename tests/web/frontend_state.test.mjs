import test from "node:test";
import assert from "node:assert/strict";
import { session_state } from "../../frontend/application.js";
const { createSession, deriveView, validatePrompt, PROMPT_LIMIT, HISTORY_LIMIT } = session_state;

const completed = { prompt: "Original opening", continuation: "Synthetic fixture text.\n\nSecond paragraph.", modelLabel: "UI test fixture", stopReason: "max_new_tokens" };

test("empty, ready and validation states do not invent output", () => {
  const session = createSession();
  assert.equal(deriveView(session.snapshot()), "empty");
  assert.equal(session.validate(), false);
  assert.equal(deriveView(session.snapshot()), "validation_error");
  session.updateDraft({ prompt: "A small story." });
  assert.equal(deriveView(session.snapshot()), "ready");
  assert.equal(session.snapshot().result, null);
  assert.deepEqual(session.snapshot().history, []);
});
test("validation counts Unicode code points and explains malformed/oversized input", () => {
  assert.match(validatePrompt(" \n\t"), /Write an opening/);
  assert.equal(validatePrompt("🦉".repeat(PROMPT_LIMIT)), "");
  assert.match(validatePrompt("🦉".repeat(PROMPT_LIMIT + 1)), /16,384/);
  assert.match(validatePrompt("a\ud800"), /invalid text character/);
  assert.match(validatePrompt(null), /as text/);
  const session = createSession();
  session.updateDraft({ prompt: "x".repeat(PROMPT_LIMIT + 1) });
  assert.equal(deriveView(session.snapshot()), "validation_error");
});
test("replacement and clear can be undone with settings intact", () => {
  const session = createSession();
  session.updateDraft({ prompt: "My opening", mode: "sample", variation: "1", length: "512" });
  const original = session.snapshot().draft;
  session.replaceDraft({ prompt: "Example" });
  session.undoReplacement();
  assert.deepEqual(session.snapshot().draft, original);
  session.replaceDraft({ prompt: "" });
  session.setFailure({ code: "invalid_request", message: "Synthetic error" });
  session.undoReplacement();
  assert.equal(session.snapshot().error, null);
  assert.deepEqual(session.snapshot().draft, original);
  assert.throws(() => session.updateDraft({ variation: 1.5 }), /Invalid draft settings/);
  assert.throws(() => session.updateDraft({ model: "../arbitrary-path" }), /Invalid draft settings/);
});
test("tab history is bounded, deduplicates openings and restores settings", () => {
  const session = createSession({ now: () => 1234 });
  assert.throws(() => session.saveOpening(), /Write an opening/);
  for (let i = 0; i < HISTORY_LIMIT + 3; i++) {
    session.updateDraft({ prompt: `Opening ${i}`, length: "256" });
    assert.equal(session.saveOpening(), true);
    assert.equal(session.saveOpening(), false);
  }
  const snapshot = session.snapshot();
  assert.equal(snapshot.history.length, HISTORY_LIMIT);
  assert.equal(snapshot.history[0].savedAt, 1234);
  session.updateDraft({ prompt: "New unsaved draft", length: "128" });
  session.restoreHistory(snapshot.history[0].id);
  assert.equal(session.snapshot().draft.prompt, "Opening 12");
  assert.equal(session.snapshot().draft.length, "256");
  session.undoReplacement();
  assert.equal(session.snapshot().draft.prompt, "New unsaved draft");
  assert.throws(() => session.restoreHistory("gone"), /no longer/);
  snapshot.history.length = 0;
  assert.equal(session.snapshot().history.length, HISTORY_LIMIT);
  assert.deepEqual(createSession().snapshot().history, []);
});
test("busy state prevents editing and output remains available after failures", () => {
  const session = createSession();
  session.updateDraft({ prompt: completed.prompt });
  session.setOutcome("success", completed);
  session.setOutcome("loading");
  assert.equal(deriveView(session.snapshot()), "loading");
  for (const action of [() => session.updateDraft({ prompt: "bad" }), () => session.replaceDraft({ prompt: "bad" }), () => session.undoReplacement(), () => session.saveOpening(), () => session.restoreHistory("entry-1")]) {
    assert.throws(action, /Wait for the current operation/);
  }
  for (const failure of ["backend_unavailable", "missing_checkpoint"]) {
    session.setOutcome(failure);
    assert.equal(deriveView(session.snapshot()), failure);
    assert.deepEqual(session.snapshot().result, completed);
  }
  session.updateDraft({ prompt: "" });
  session.validate();
  assert.equal(deriveView(session.snapshot()), "validation_error");
  assert.equal(session.snapshot().result.prompt, completed.prompt);
});
test("completed output is associated with its original prompt and handles immediate EOS", () => {
  const session = createSession();
  session.updateDraft({ prompt: "New draft" });
  session.setOutcome("success", completed);
  session.restoreHistory(session.snapshot().history[0].id);
  assert.equal(session.snapshot().draft.prompt, completed.prompt);
  assert.deepEqual(session.snapshot().result, completed);
  session.setOutcome("success", { ...completed, continuation: "", stopReason: "eos" });
  assert.equal(deriveView(session.snapshot()), "success");
  assert.throws(() => session.setOutcome("success", { ...completed, stopReason: "invented" }), /Invalid completed output/);
});
test("subscriptions release and snapshots cannot mutate internal state", () => {
  const session = createSession();
  let notifications = 0;
  const unsubscribe = session.subscribe(snapshot => { notifications++; snapshot.draft.prompt = "mutated"; });
  session.updateDraft({ prompt: "Real draft" });
  assert.equal(session.snapshot().draft.prompt, "Real draft");
  unsubscribe();
  session.updateDraft({ prompt: "Another" });
  assert.equal(notifications, 1);
});

test("extended model starts with sampling while an explicit selection survives refresh", () => {
  const session = createSession();
  const models = [
    { model_id: "baseline", kind: "baseline", availability: "registered" },
    { model_id: "extended", kind: "english", availability: "registered" },
  ];
  const catalog = { models, registry_fingerprint: "fixture" };
  session.setCatalog(catalog);
  const { model, mode, variation, topK } = session.snapshot().draft;
  assert.deepEqual({ model, mode, variation, topK }, {
    model: "extended", mode: "sample", variation: "0.65", topK: "10",
  });
  session.updateDraft({ model: "baseline", mode: "greedy" });
  session.setCatalog(catalog);
  assert.equal(session.snapshot().draft.model, "baseline");
  assert.equal(session.snapshot().draft.mode, "greedy");
});

test("a fixed-model session keeps its generation model when history is reopened", () => {
  const session = createSession({ fixedModel: "combined" });
  session.recordGeneration({ ...session.snapshot().draft, model: "baseline" }, completed);
  session.restoreHistory(session.snapshot().history[0].id);
  assert.equal(session.snapshot().draft.model, "combined");
  assert.equal(session.snapshot().result.modelLabel, completed.modelLabel);
});
