import test from "node:test";
import assert from "node:assert/strict";
import { session_state } from "../../frontend/application.js";
const { createSession } = session_state;
import { generation_client } from "../../frontend/application.js";
const { createGenerationClient, validateCatalog, validateResult, requestFromDraft } = generation_client;

// Synthetic transport responses belong only in this offline test file.
const catalog = { registry_fingerprint: "c".repeat(64), models: [{ model_id: "fixture-baseline-v1", kind: "baseline", display_name: "Synthetic test model", availability: "registered", unavailable_reason: null, checkpoint: { sha256: "a".repeat(64), update: 1 }, tokenizer_fingerprint: "b".repeat(64) }] };
const status = { phase: 15, contract_version: "2.0.0", generation_enabled: true, busy: false };
const response = (body, ok = true) => ({ ok, json: async () => body });
function fixtureResult(request) {
  return { request_id: "d".repeat(32), prompt: request.prompt, continuation: "Synthetic test output.", model_id: request.model_id, model_kind: "baseline", checkpoint: catalog.models[0].checkpoint,
    sampling_settings: { ...request.settings, temperature: request.settings.temperature ?? 0, top_k: null, seed: null, effective_seed: request.settings.mode === "sample" ? "9223372036854775807" : null, stop_at_eos: true },
    generated_token_count: 22, stop_reason: "max_new_tokens", generation_duration_seconds: .1, model_load_seconds: .2, initial_prompt_tokens_truncated: 0, unknown_prompt_tokens: 0, context_length: 256,
    tokenizer_fingerprint: catalog.models[0].tokenizer_fingerprint, registry_fingerprint: catalog.registry_fingerprint, device: "cpu", cache_hit: false };
}
function ready() {
  const session = createSession();
  session.setCatalog(catalog);
  session.updateDraft({ prompt: "A small story." });
  return session;
}

test("registry refresh and repeated view subscriptions never submit generation", async () => {
  const session = createSession();
  const calls = [];
  const client = createGenerationClient(session, { fetcher: async (url, options) => {
    calls.push([url, options.method]);
    return response(url.endsWith("status") ? status : catalog);
  } });
  await client.refresh();
  for (let i = 0; i < 4; i++) { const unsubscribe = session.subscribe(() => {}); session.updateDraft({ prompt: "Draft" }); unsubscribe(); }
  assert.deepEqual(calls, [["/api/v1/status", undefined], ["/api/v1/models", undefined]]);
  assert.equal(session.snapshot().draft.model, "fixture-baseline-v1");
  assert.equal(session.snapshot().connection, "ready");
});

test("duplicate submission is ignored; completion uses its snapshot and exact model identity", async () => {
  const session = ready();
  let release, calls = 0;
  const client = createGenerationClient(session, { fetcher: async (url, options) => {
    assert.equal(url, "/api/v1/generations");
    assert.equal(options.method, "POST");
    assert.equal(options.credentials, "omit");
    calls++;
    const payload = JSON.parse(options.body);
    await new Promise(resolve => { release = resolve; });
    return response(fixtureResult(payload));
  } });
  const pending = client.submit();
  assert.equal(session.snapshot().outcome, "loading");
  await client.submit();
  assert.equal(calls, 1);
  assert.throws(() => session.updateDraft({ prompt: "Wrong prompt" }), /Wait/);
  release();
  await pending;
  assert.equal(session.snapshot().result.prompt, "A small story.");
  assert.equal(session.snapshot().result.metadata.checkpoint.sha256, catalog.models[0].checkpoint.sha256);
  assert.equal(session.snapshot().history.length, 1);
});

test("server error keeps prior output and prompt; editing recovers and retry succeeds", async () => {
  const session = ready();
  let fail = false;
  const client = createGenerationClient(session, { fetcher: async (_, options) => fail
    ? response({ error: { code: "invalid_request", message: "Remove unknown character U+2603.", field: "prompt" } }, false)
    : response(fixtureResult(JSON.parse(options.body))) });
  await client.submit();
  const previous = session.snapshot().result;
  session.updateDraft({ prompt: "Changed opening ☃" }); fail = true;
  await client.submit();
  assert.deepEqual(session.snapshot().result, previous);
  assert.equal(session.snapshot().draft.prompt, "Changed opening ☃");
  assert.equal(session.snapshot().error.field, "prompt");
  session.updateDraft({ prompt: "Changed opening" }); fail = false;
  assert.equal(session.snapshot().error, null);
  await client.submit();
  assert.equal(session.snapshot().result.prompt, "Changed opening");
});

test("busy, artifact and network failures are explicit and never automatically retried", async () => {
  for (const code of ["busy", "artifact_unavailable", "network"]) {
    const session = ready(); let calls = 0;
    const client = createGenerationClient(session, { fetcher: async () => {
      calls++;
      if (code === "network") throw new Error("offline");
      return response({ error: { code, message: "Test recovery instructions.", field: "model_id" } }, false);
    } });
    await client.submit();
    assert.equal(calls, 1);
    assert.equal(session.snapshot().draft.prompt, "A small story.");
    assert.equal(session.snapshot().error.code, code === "network" ? "service_unavailable" : code);
    assert.equal(session.snapshot().result, null);
    if (code === "artifact_unavailable") assert.equal(session.snapshot().models[0].availability, "unavailable");
  }
});

test("settings map the visible controls without sending paths or greedy sampling fields", () => {
  const session = ready();
  assert.deepEqual(requestFromDraft(session.snapshot().draft).settings, { mode: "greedy", max_new_tokens: 128, unknown_policy: "error" });
  session.updateDraft({ mode: "sample", variation: "1", length: "512" });
  const request = requestFromDraft(session.snapshot().draft);
  assert.equal(request.settings.temperature, 1);
  const displayed = validateResult(fixtureResult(request), request, catalog);
  assert.equal(displayed.metadata.sampling_settings.effective_seed, "9223372036854775807");
  assert.throws(() => requestFromDraft({ ...session.snapshot().draft, variation: "NaN" }), /Temperature/);
});

test("mismatched checkpoint, tokenizer, prompt, settings and malformed metadata are rejected", () => {
  const request = requestFromDraft(ready().snapshot().draft);
  for (const changes of [{ checkpoint: { sha256: "e".repeat(64), update: 1 } }, { tokenizer_fingerprint: "f".repeat(64) }, { prompt: "Wrong" }, { model_id: "other" }, { generation_duration_seconds: NaN }, { generated_token_count: 129 }, { sampling_settings: { mode: "sample" } }]) {
    assert.throws(() => validateResult({ ...fixtureResult(request), ...changes }, request, catalog), /did not match/);
  }
  assert.throws(() => validateCatalog({ ...catalog, models: [catalog.models[0], catalog.models[0]] }), /Invalid model/);
  assert.throws(() => validateCatalog({ ...catalog, models: [{ ...catalog.models[0], model_id: undefined }] }), /Invalid model/);
});

test("independent browser sessions retain independent prompts and histories", async () => {
  const a = ready(), b = ready(); b.updateDraft({ prompt: "Second tab" });
  const client = createGenerationClient(a, { fetcher: async (_, options) => response(fixtureResult(JSON.parse(options.body))) });
  await client.submit();
  assert.equal(a.snapshot().history.length, 1);
  assert.equal(b.snapshot().draft.prompt, "Second tab");
  assert.equal(b.snapshot().history.length, 0);
  assert.equal(b.snapshot().result, null);
});

test("status timeout and malformed responses preserve the draft", async () => {
  const session = ready();
  const client = createGenerationClient(session, { statusTimeoutMs: 5, fetcher: (_, { signal }) => new Promise((resolve, reject) => signal.addEventListener("abort", () => reject(new Error("timeout")), { once: true })) });
  await client.refresh();
  assert.equal(session.snapshot().connection, "unreachable");
  assert.equal(session.snapshot().draft.prompt, "A small story.");
  const incompatible = createGenerationClient(session, { fetcher: async () => response({ phase: 14 }) });
  await incompatible.refresh();
  assert.equal(session.snapshot().connection, "unreachable");
});
