---
title: CoDA-GQA-L Neural Database
emoji: "\U0001F9E0"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
pinned: false
license: mit
suggested_hardware: a10g-small
---

# CoDA-GQA-L: Stateful Neural Database

Process documents into fixed-size neural states (~61 MB).
Save them to a small state library, reload them later, and query without
re-reading the original document.

The app persists saved states under `STATE_LIBRARY_DIR` (defaults to
`demo/state_library`) and exposes both one-off `.pt` uploads and a
named in-app library of saved document states.

See [CoDA-GQA-L](https://github.com/anthony-maio/CoDA-GQA-L) for details.
