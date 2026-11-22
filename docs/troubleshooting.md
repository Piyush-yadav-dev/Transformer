# Troubleshooting

| Symptom | Next step |
| --- | --- |
| .venv/bin/python: no such file | Change to the actual project directory first; create the isolated environment using README installation if absent. |
| transformer import missing after edits | Rerun locked uv sync with --no-editable --reinstall-package transformer. |
| Web module/dependency missing | Include --extra web (and --extra evaluation for plots); syncing without extras can remove them. |
| MPS unavailable | Run diagnose --device auto; use CPU explicitly. Do not report CPU checks as MPS verification. |
| Checkpoint/tokenizer missing | Restore the trusted artifact archive or train/register a new version; fixtures do not replace trained models. |
| SHA or metadata mismatch | Restore matching original files. Do not edit registry hashes to accept unexplained changes. |
| Run/output directory exists | Use a new ID/output for a new run, or resume the exact existing run with --resume. |
| Historical resume rejected | Restore recorded code/configuration/environment; see checkpoint recovery. |
| Unknown prompt character | Remove/replace the reported character; shared vocabulary is frozen. |
| Invalid top-k/temperature | Use positive temperature and integer top-k no larger than vocabulary; greedy omits sampling-only fields. |
| Busy generation | Wait for the active job. Refresh does not cancel accelerator work. |
| Empty analytics or warning | Inspect source evidence and local files; missing metrics are not zero. Restore original evaluation files. |
| Local port occupied | Use --port 8002 or another free allowed port; do not stop unrelated processes. |
| History disappeared | Reload clears tab-local history by design; navigation within the app preserves it. |

Technical web errors are recorded in logs/web/inference.jsonl. Avoid sharing logs
without reviewing contents. No streaming/cancel button is implemented. Sampling seed
repeatability is limited to compatible runtime/device conditions.

Verification gaps are recorded in [usability review](phase18_usability.md), not
hidden by later passing backend tests.
