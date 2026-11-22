# Maintenance commands

Run `.venv/bin/python tools/maintenance.py --help` from the project root.
Use a listed subcommand, followed by its options. For example:

```sh
.venv/bin/python tools/maintenance.py preview-ui-fixtures --port 8001
.venv/bin/python tools/maintenance.py export-story-charts
```

These commands are not needed to launch the website. Historical verification and
benchmark commands require the experiment artifacts named in their documentation;
they do not replace a missing experiment with the currently registered model.
