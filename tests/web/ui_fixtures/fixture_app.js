import { session_state } from "/assets/application.js";
const { createSession, EXAMPLE_PROMPTS, PROMPT_LIMIT } = session_state;
import { generate_page } from "/assets/application.js";
const { mountGeneratePage } = generate_page;

// Deliberately outside the installable package: these are not generated stories.
const longOutput = [
  "SYNTHETIC UI TEST FIXTURE — not a model response. This paragraph checks comfortable line length and selectable text.",
  ...Array.from({ length: 6 }, (_, i) => `Fixture paragraph ${i + 1}. ` + "This intentionally repeated sentence checks paragraph spacing, wrapping and the ability to read a long continuation. ".repeat(5)),
  "Literal markup must stay text: <script>alert('fixture')</script> & <b>not bold</b>.",
  "Long unbroken fixture text: " + "abcdefghij".repeat(40),
].join("\n\n");
const response = await fetch("/assets/index.html");
if (!response.ok) throw new Error("The shared Generate template could not be loaded.");
const template = new DOMParser().parseFromString(await response.text(), "text/html").querySelector("#generate-page");
const root = document.querySelector("#fixture-root");
const selector = document.querySelector("#scenario");
let unmount = () => {};
function showScenario() {
  unmount();
  const session = createSession();
  session.setCatalog({ registry_fingerprint: "f".repeat(64), models: [
    { model_id: "fixture-baseline-v1", kind: "baseline", display_name: "UI test fixture", availability: "registered", unavailable_reason: null, checkpoint: { sha256: "a".repeat(64), update: 1 }, tokenizer_fingerprint: "b".repeat(64) },
  ] });
  session.setConnection("not_connected");
  const scenario = selector.value;
  if (!["empty", "validation_error"].includes(scenario)) session.updateDraft({ prompt: EXAMPLE_PROMPTS[0].text });
  if (scenario === "validation_error") session.validate();
  if (scenario === "backend_unavailable") session.setConnection("unreachable");
  if (["loading", "missing_checkpoint"].includes(scenario)) session.setOutcome(scenario);
  if (["success", "immediate_eos"].includes(scenario)) session.setOutcome("success", {
    prompt: session.snapshot().draft.prompt,
    continuation: scenario === "success" ? longOutput : "",
    modelLabel: "UI test fixture · synthetic text",
    stopReason: scenario === "success" ? "max_new_tokens" : "eos",
  });
  if (["long_prompt", "oversized_prompt"].includes(scenario)) {
    const size = PROMPT_LIMIT + (scenario === "oversized_prompt" ? 1 : 0);
    session.updateDraft({ prompt: "Synthetic prompt fixture. ".repeat(700).slice(0, size) });
  }
  root.replaceChildren(template.content.cloneNode(true));
  unmount = mountGeneratePage(root, session);
  document.querySelector("#fixture-status").textContent = `Fixture: ${selector.selectedOptions[0].textContent}. No inference request will be sent.`;
}
selector.addEventListener("change", showScenario);
showScenario();
