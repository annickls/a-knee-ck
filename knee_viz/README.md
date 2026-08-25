# knee_viz

Live visualisation of tibiofemoral kinematics: the two registered bones in 3D on
the left, and the selected quantity plotted against flexion angle on the right.

This replaces the "bone visualization" tab of `GUI_with_bones_lets_go.py`. That
file is untouched and still runs; nothing here imports from it.

## Running

```bash
# Replay a finished recording (demo / development)
python -m knee_viz --source replay --csv data_20260729_175019.csv --hz 60

# Follow a CSV that the ROS 2 recorder is appending to (live)
python -m knee_viz --source tail --csv ../knee_eval_ws/data_latest.csv

# Diagnostic overlay: landmarks, anatomical axes, joint-gap rays
python -m knee_viz --source replay --csv ... --debug
```

Options: `--model-dir` (defaults to `data_for_gui/Model_demo`), `--side left|right`,
`--no-loop`. `KNEEVIZ_DEBUG=1` is equivalent to `--debug`.

In the window, `D` toggles the debug overlay and `R` recentres the camera.

## Install

```bash
pip install -r requirements-knee_viz.txt
sudo apt install libxcb-cursor0      # PyQt6 >= 6.5 needs this to open a window
```

## Layout

`core` and `data` import no Qt, so the whole measurement chain runs headless and
is covered by tests without constructing a widget.

```
config.py     frozen AppConfig, built once from the CLI and injected
theme.py      the only file containing a colour literal

core/         transforms  quaternion/pose maths; the sole metres -> mm conversion
              assets      .fcsv, marker .yaml and STL readers
              registration Kabsch; world = R @ stl + t
              bones       BoneModel: mesh + landmarks, registration baked in at load
              kinematics  Grood & Suntay: KneeFrames, Dof6, ZeroOffsets
              joint_gap   ray-cast gap along the tibial anatomical axis
              modes       plot modes and their neutrality gates
              pipeline    Frame -> KneeState, once per frame

data/         source      Frame + DataSource contract
              csv_format  the column layout; nothing else knows indices
              csv_tail    follows a file being appended to
              csv_replay  plays a finished file at a set rate
              driver      QTimer -> poll -> frameReady   <- the Qt boundary

ui/           main_window, scene3d, plot2d, readouts, controls,
              actors, debug_layer, overlays, shading, force_bar
```

## How the measurement works

Each bone's STL and landmarks are Kabsch-registered into that bone's tracker
reference frame **once at load**, so per frame only a 4x4 pose is applied and the
geometry arrays never move.

**Joint gap.** A ray is cast from each tibial plateau centre landmark
(`tibia_medial`, `tibia_lateral`) along the tibial anatomical axis
`e3t = normalize(tibia_proximal - tibia_distal)`, and the gap is the distance to
the first intersection with the femoral surface. The rays are transformed into
the static femur frame rather than the femur being transformed, which allows the
femoral triangles to be cropped to the condylar region once at construction.
Per frame, triangles further from the ray than their own circumradius are
rejected before any intersection test runs. This is exact: the result is
bit-identical to Möller–Trumbore over all 164528 triangles, at ~1 ms per ray.

**Zeroing** subtracts a captured baseline from all six degrees of freedom after
they are solved, so no offset can be overwritten by a later assignment.

**The neutrality gate.** A sample only counts for the quantity being plotted if
the *other* degrees of freedom are near neutral — a varus/valgus reading taken
while the tibia is rotated 20 degrees is not comparable with one taken in
neutral rotation. Gated-out samples are drawn dim; passing samples are green.
Each mode declares its own gate in `core/modes.py`.

## Tests

```bash
python -m pytest          # 52 tests, ~2 s
```

`pytest.ini` disables the system ROS 2 pytest plugins, which are registered in
this interpreter and fail to import.

Beyond the unit tests, `tests/test_structure.py` enforces two invariants:
`core` and `data` must import no GUI library, and colour literals must not
appear outside `theme.py`.

## Extending

- **Force bar** — `ui/force_bar.py` is built, laid out and called every frame with
  `set_value(None)`. `Frame.force_n` is already parsed. Wiring it up is one line
  in `MainWindow._on_frame` plus `AppConfig.show_force_bar = True`.
- **A new plot mode** — add one entry to `MODES` in `core/modes.py`. The button,
  axis limits, captions and gate all follow from it.
- **A new data source** — implement `DataSource` and add it to `make_source`.

## Known issue: Model_demo registration

`data_for_gui/Model_demo/marker_coordinates.yaml` does not match the marker
mounting used for the recordings in the repository root. With that pairing the
femur and tibia sit roughly 150–230 mm apart instead of articulating, so the
angles and gaps it produces are not meaningful.

This is a data issue, not a code one, and it affects the legacy GUI identically —
pinned to the same row, the old GUI reports a 231 mm condyle-to-plateau
separation and translations of −134 / −142 / −117 mm.

The assets are internally consistent (marker fit RMSD 0.0045 mm; the femur and
tibia STLs articulate correctly in the CT frame), and the same code gives
0.3–9.0 mm bone apposition on seven of the eight `P*_pre` folders. Use
`--model-dir data_for_gui/P6_pre` for a correctly registered example. Fixing
`Model_demo` means re-digitising the `*_slicer` marker points against the current
physical mounting.
