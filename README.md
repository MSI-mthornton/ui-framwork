# MSI seismic UI framework

A shared base for the 6-8 Qt apps that display gathers for detected
microseismic events. It replaces the per-app copies of threading, caching,
rotation and plotting code in qc-keepers and event3CQC with two packages and
a reference app, all in one [uv](https://docs.astral.sh/uv/) workspace.

```
packages/
  seiscore/   Qt-free core: Event/EventTable/Gather, conventions, DSP, config,
              data sources (Fake, PSET files, Pulse)
  seisqt/     PySide6 + pyqtgraph: task runner, gather cache, stores,
              controllers, base window, panels
apps/
  demo-viewer/  reference app on synthetic data (no Pulse needed)
```

## Quick start

```bash
uv sync                      # creates .venv with all packages + dev tools
uv run demo-viewer           # opens the demo app
uv run pytest                # tests (set QT_QPA_PLATFORM=offscreen on a headless box)
uv run ruff check . && uv run ruff format --check .
```

Pulse support needs `pulse-sdk` and `pulse-schemas`, which come from the private
repo, so they are not declared as dependencies. Add them to the app that uses
Pulse (for example `uv add --package my-app "pulse-sdk @ git+ssh://..."`).

## Layering rules

1. **seiscore never imports Qt.** Anything testable without a display lives here:
   domain types, rotation, sorting, clip levels, stacks, decimation, file and
   Pulse readers, config.
2. **Panels never call sources.** Panels read stores and call controllers.
   Controllers call sources through the `TaskRunner`, so every slow call runs on
   a worker thread and reports back as a `Result` on the GUI thread. A failure
   or a timeout is a `Result`, never a hang.
3. **Data is immutable.** `EventTable.with_qc` and `Gather.take_stations` return
   new objects, so workers and the GUI never race on one array, and caching is
   safe.
4. **Process at full rate, decimate only for display.** Stacks and clip levels
   use every sample; `decimate_for_display` (min/max, keeps signed peaks) runs
   last, on a worker, sized to the panel.

## Conventions

Defined once in `seiscore.conventions`, and converted at the source boundary:

- Coordinates: x = east, y = north, z = **up**, metres.
- Raw components `E`, `N`, `Z` with Z up (PSET trid 12 is down and is negated
  on read).
- Ray frame: **P** = unit vector source to receiver, **SH** = unit(z × P),
  **SV** = SH × P. This matches qc-keepers. event3CQC pointed P the other way,
  so its P panel had the opposite polarity.
- `Gather.data` is `(component, station, sample)`; sample k is at
  `t0 + k*dt` seconds relative to `reference` (`"origin"` or `"arrival:P"`).

## Sizing

Projects have 1e3-1e5 events and 1e2-5e3 stations. What that drove:

- Event scatters group points by QC code (one uniform brush each) and draw the
  selection as a separate overlay, so moving the selection never redraws 1e5
  points, and a QC change rebuilds only the two affected groups.
- A full-rate 5e3-station gather is a few hundred MB. The cache is bounded in
  **bytes** (`GatherCache(max_bytes=...)`), rotation runs in station blocks, and
  display preparation runs off the GUI thread.
- PSET files are memory-mapped and indexed by time span; a gather read is a
  slice per file, not a loop over traces.

## Threading

`TaskRunner.submit(fn, on_done, key=..., timeout=...)`:

- **key**: a newer submission with the same key supersedes the older one (its
  result is dropped). Fast scrolling through events never queues stale redraws.
- **timeout**: the callback fires with `timed_out=True`; a late result is
  discarded. Python threads cannot be killed, so the worker still finishes.
- Sources must be thread-safe. The sync `PulseClient` is; the async one probably
  is not, so `seiscore.sources.pulse` shares one sync client.

## Making a new app

Copy `apps/demo-viewer` and rename it. An app supplies three things:

```python
class MyApp(BaseApp):
    name = "my-app"
    config_cls = MyConfig              # frozen dataclasses, one per TOML table
    packaged_config = PACKAGED         # defaults.toml next to the code
    env_prefix = "MYAPP"               # MYAPP_PULSE__HOST=... overrides

    def sources(self, config):         # where data comes from
        conn, sources = pulse_sources(config.pulse, app_name=self.name)
        return sources

    def build(self, ctx):              # which panels and commands
        win = BaseMainWindow(ctx)
        win.set_central(GatherPanel(ctx))
        win.add_panel(EventMapPanel(ctx), "Map")
        return win
```

Then add it to the root `pyproject.toml` (`dependencies` and
`[tool.uv.sources]`) and run `uv sync`. The base app gives you `--project
file.toml`, `--set key=value`, `--log-dir`, `--debug`, a display check with
fixes for the usual X11 problems, logging to a panel and file, an excepthook
that logs instead of crashing, remembered window layout and view settings, and
standard event navigation shortcuts.

Config precedence, lowest to highest: dataclass defaults, packaged TOML,
project TOML, environment, `--set`. Unknown keys are errors, and relative paths
resolve against the project file.

## Data sources

| Source | Events | Gathers | QC writes | Status |
| --- | --- | --- | --- | --- |
| `sources.fake` | yes | yes (P/S Ricker pulses, physically polarised) | records | tested |
| `sources.pset` | - | yes (origin or arrival aligned, needs a travel-time function for arrival) | - | tested against files written by `write_pset`; writer checked against event3CQC's `seistrc` reader |
| `sources.pulse` | yes (drops superseded PSET runs) | yes (full-rate, rotated in seiscore) | yes | tested with a stand-in client; **not yet run against a live Pulse server** |

## Windows notes

Everything is pure Python. `uv sync` installs PySide6 wheels on Windows with no
extra system packages. On Linux servers, `seisqt.diagnostics` prints the `dnf`
line for missing Qt libraries and explains DISPLAY / xauth failures.
