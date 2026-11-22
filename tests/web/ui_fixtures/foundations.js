import { components } from "/assets/application.js";
const { renderState } = components;

function mountFoundations() {
  const specimens = document.querySelector("#state-specimens");
  for (const [kind, title, description] of [
    ["loading", "Loading example", "A real request would show progress here, without an invented percentage."],
    ["empty", "Empty example", "An honest empty state explains what will appear and how to begin."],
    ["success", "Success example", "Confirmation follows a completed action. No model result is being shown."],
    ["error", "Error example", "A useful error preserves the draft and explains the next step."],
  ]) {
    const wrapper = document.createElement("div");
    specimens.append(wrapper);
    renderState(wrapper, { kind, title, description });
  }
  const feedback = document.querySelector("#sample-feedback");
  document.querySelector("#sample-action").addEventListener("click", () => renderState(feedback, {
    kind: "success", title: "Feedback example displayed.", description: "This is an interface specimen, not a generation result.",
  }));
  document.querySelector("#sample-reset").addEventListener("click", () => feedback.replaceChildren());
}

mountFoundations();
