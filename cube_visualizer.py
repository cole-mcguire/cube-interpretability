"""Interactive tkinter visualizer for the 2x2x2 cube simulator."""

from __future__ import annotations

import math
import os
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np # pyright: ignore[reportMissingImports]

from cube import Cube, DistanceTable, IDX_TO_COLOR, MOVE_NAMES, solve


def _find_project_root() -> str:
    """Walk up from CWD until we find a directory containing pyproject.toml."""
    d = os.getcwd()
    for _ in range(6):
        if os.path.exists(os.path.join(d, "pyproject.toml")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.getcwd()

_PROJECT_ROOT = _find_project_root()
_DISTANCES_CACHE = os.path.join(_PROJECT_ROOT, "data", "distances.npz")


FACE_ORDER = ["U", "D", "F", "B", "L", "R"]
FACE_START = {name: idx * 4 for idx, name in enumerate(FACE_ORDER)}
FACE_GRID_POSITIONS = {
    "U": (2, 0),
    "L": (0, 2),
    "F": (2, 2),
    "R": (4, 2),
    "B": (6, 2),
    "D": (2, 4),
}
COLOR_TO_HEX = {
    "W": "#f8fafc",
    "Y": "#facc15",
    "O": "#fb923c",
    "R": "#ef4444",
    "G": "#22c55e",
    "B": "#3b82f6",
}
STICKER_SIZE = 32
STICKER_GAP = 4
FACE_GAP = 10
MARGIN_X = 20
MARGIN_Y = 24
CONTROL_PANEL_WIDTH = 340


# ---------------------------------------------------------------------------
# 3D geometry
# ---------------------------------------------------------------------------

_FACE_AXES_3D = {
    "U": {"normal": (0, 1, 0),  "right": (1, 0, 0),  "down": (0, 0, 1)},
    "D": {"normal": (0, -1, 0), "right": (1, 0, 0),  "down": (0, 0, -1)},
    "F": {"normal": (0, 0, 1),  "right": (1, 0, 0),  "down": (0, -1, 0)},
    "B": {"normal": (0, 0, -1), "right": (-1, 0, 0), "down": (0, -1, 0)},
    "L": {"normal": (-1, 0, 0), "right": (0, 0, 1),  "down": (0, -1, 0)},
    "R": {"normal": (1, 0, 0),  "right": (0, 0, -1), "down": (0, -1, 0)},
}


def _build_sticker_quads() -> list[tuple[tuple, list]]:
    """Return one (normal, [corner0..corner3]) tuple per sticker, in cube.py order."""
    S = 0.44  # half-sticker size; leaves a ~0.12-unit gap between stickers
    quads = []
    for face_name in ["U", "D", "F", "B", "L", "R"]:
        ax = _FACE_AXES_3D[face_name]
        n, r, d = ax["normal"], ax["right"], ax["down"]
        for row in range(2):
            for col in range(2):
                rs = -1 if row == 0 else 1
                cs = -1 if col == 0 else 1
                cx = n[0] + cs * 0.5 * r[0] + rs * 0.5 * d[0]
                cy = n[1] + cs * 0.5 * r[1] + rs * 0.5 * d[1]
                cz = n[2] + cs * 0.5 * r[2] + rs * 0.5 * d[2]
                # corners in CCW order as seen from outside
                corners = []
                for (dr, dc) in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
                    corners.append((
                        cx + dc * S * r[0] + dr * S * d[0],
                        cy + dc * S * r[1] + dr * S * d[1],
                        cz + dc * S * r[2] + dr * S * d[2],
                    ))
                quads.append((n, corners))
    return quads


_STICKER_QUADS = _build_sticker_quads()


# ---------------------------------------------------------------------------
# 3D canvas widget
# ---------------------------------------------------------------------------

class Cube3DCanvas(ttk.Frame):
    SIZE = 260
    SCALE = 88

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self.yaw = math.radians(35)
        self.pitch = math.radians(22)
        self._drag_start: tuple | None = None
        self._last_state: np.ndarray | None = None

        self.canvas = tk.Canvas(
            self,
            width=self.SIZE,
            height=self.SIZE,
            background="#111827",
            highlightthickness=0,
        )
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)

    def _on_press(self, event: tk.Event) -> None:
        self._drag_start = (event.x, event.y, self.yaw, self.pitch)

    def _on_drag(self, event: tk.Event) -> None:
        if self._drag_start is None or self._last_state is None:
            return
        x0, y0, yaw0, pitch0 = self._drag_start
        self.yaw = yaw0 + (event.x - x0) * 0.01
        self.pitch = max(
            -math.pi / 2 + 0.05,
            min(math.pi / 2 - 0.05, pitch0 + (event.y - y0) * 0.01),
        )
        self.redraw(self._last_state)

    def _rotate(self, v: tuple) -> tuple:
        x, y, z = v
        x2 = x * math.cos(self.yaw) + z * math.sin(self.yaw)
        z2 = -x * math.sin(self.yaw) + z * math.cos(self.yaw)
        y3 = y * math.cos(self.pitch) - z2 * math.sin(self.pitch)
        z3 = y * math.sin(self.pitch) + z2 * math.cos(self.pitch)
        return x2, y3, z3

    def _project(self, v: tuple) -> tuple:
        rx, ry, rz = self._rotate(v)
        cx = self.SIZE / 2 + rx * self.SCALE
        cy = self.SIZE / 2 - ry * self.SCALE
        return cx, cy, rz

    def redraw(self, cube_state: np.ndarray) -> None:
        self._last_state = cube_state
        self.canvas.delete("all")

        visible: list[tuple] = []
        for sticker_idx, (normal, corners_3d) in enumerate(_STICKER_QUADS):
            _, _, nz = self._rotate(normal)
            if nz <= 0:
                continue  # backface culling
            proj = [self._project(c) for c in corners_3d]
            depth = sum(p[2] for p in proj) / 4
            pts_2d = [(p[0], p[1]) for p in proj]
            color_name = IDX_TO_COLOR[int(cube_state[sticker_idx])]
            visible.append((depth, pts_2d, COLOR_TO_HEX[color_name]))

        for _, pts, color in sorted(visible, key=lambda x: x[0]):
            flat = [c for pt in pts for c in pt]
            self.canvas.create_polygon(flat, fill=color, outline="#111827", width=3)


class CubeVisualizer:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("2x2x2 Cube Visualizer")
        self.cube = Cube()
        self.rng = np.random.default_rng()
        self.move_history: list[str] = []

        self.status_var = tk.StringVar()
        self.history_var = tk.StringVar()
        self.scramble_length_var = tk.IntVar(value=12)

        self._distances: dict | None = None
        self._distances_loading = False
        self._solution_moves: list[int] = []
        self._solution_step = 0
        self._playing = False
        self._play_delay_var = tk.IntVar(value=500)
        self._solve_status_var = tk.StringVar()

        self._build_ui()
        self._bind_shortcuts()
        self.redraw()
        self.cube3d.redraw(self.cube.state)

    def _build_ui(self) -> None:
        canvas_width, canvas_height = self._canvas_dimensions()
        self.root.minsize(canvas_width + CONTROL_PANEL_WIDTH + 72, canvas_height + 64)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        frame = ttk.Frame(self.root, padding=16)
        frame.grid(sticky="nsew")
        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            frame,
            width=canvas_width,
            height=canvas_height,
            background="#111827",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nw", padx=(0, 16))

        self.cube3d = Cube3DCanvas(frame)
        self.cube3d.grid(row=1, column=0, sticky="nw", padx=(0, 16), pady=(12, 0))

        controls = ttk.Frame(frame)
        controls.grid(row=0, column=1, sticky="nsew")
        controls.columnconfigure(0, weight=1)

        ttk.Label(
            controls,
            text="Moves",
            font=("Helvetica", 14, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        move_grid = ttk.Frame(controls)
        move_grid.grid(row=1, column=0, sticky="ew")
        for col in range(3):
            move_grid.columnconfigure(col, weight=1)

        for idx, move_name in enumerate(MOVE_NAMES):
            button = ttk.Button(
                move_grid,
                text=move_name,
                command=lambda name=move_name: self.apply_move(name),
            )
            button.grid(row=idx // 3, column=idx % 3, sticky="ew", padx=2, pady=2)

        actions = ttk.Frame(controls)
        actions.grid(row=2, column=0, sticky="ew", pady=(16, 0))
        actions.columnconfigure(1, weight=1)

        ttk.Button(actions, text="Reset", command=self.reset_cube).grid(
            row=0, column=0, sticky="ew", padx=(0, 8)
        )
        ttk.Label(actions, text="Scramble").grid(row=0, column=1, sticky="w")
        ttk.Spinbox(
            actions,
            from_=1,
            to=100,
            textvariable=self.scramble_length_var,
            width=5,
        ).grid(row=0, column=2, sticky="w", padx=(8, 8))
        ttk.Button(actions, text="Run", command=self.scramble_cube).grid(
            row=0, column=3, sticky="ew"
        )

        ttk.Label(
            controls,
            text="Keyboard: u d f b l r for clockwise, uppercase for inverse turns.",
            wraplength=320,
            foreground="#4b5563",
        ).grid(row=3, column=0, sticky="w", pady=(16, 10))

        ttk.Label(
            controls,
            textvariable=self.status_var,
            wraplength=320,
            justify="left",
        ).grid(row=4, column=0, sticky="w")

        ttk.Label(
            controls,
            textvariable=self.history_var,
            wraplength=320,
            justify="left",
        ).grid(row=5, column=0, sticky="w", pady=(12, 0))

        ttk.Separator(controls, orient="horizontal").grid(
            row=6, column=0, sticky="ew", pady=(16, 12)
        )

        ttk.Label(
            controls,
            text="Solver",
            font=("Helvetica", 14, "bold"),
        ).grid(row=7, column=0, sticky="w", pady=(0, 8))

        solver_top = ttk.Frame(controls)
        solver_top.grid(row=8, column=0, sticky="ew")
        solver_top.columnconfigure(1, weight=1)

        self._solve_btn = ttk.Button(
            solver_top, text="Solve", command=self._on_solve
        )
        self._solve_btn.grid(row=0, column=0, sticky="w", padx=(0, 8))

        ttk.Label(
            solver_top,
            textvariable=self._solve_status_var,
            wraplength=240,
            justify="left",
            foreground="#4b5563",
        ).grid(row=0, column=1, sticky="w")

        solver_bottom = ttk.Frame(controls)
        solver_bottom.grid(row=9, column=0, sticky="ew", pady=(8, 0))

        self._step_btn = ttk.Button(
            solver_bottom, text="Step", command=self._on_step, state="disabled"
        )
        self._step_btn.grid(row=0, column=0, sticky="w", padx=(0, 4))

        self._play_btn = ttk.Button(
            solver_bottom, text="▶ Play", command=self._on_play, state="disabled"
        )
        self._play_btn.grid(row=0, column=1, sticky="w", padx=(0, 12))

        ttk.Label(solver_bottom, text="Speed:").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(
            solver_bottom,
            from_=50,
            to=2000,
            increment=50,
            textvariable=self._play_delay_var,
            width=5,
        ).grid(row=0, column=3, sticky="w", padx=(4, 4))
        ttk.Label(solver_bottom, text="ms").grid(row=0, column=4, sticky="w")

    def _bind_shortcuts(self) -> None:
        for face in "UDFBLR":
            self.root.bind(face.lower(), lambda event, f=face: self.apply_move(f))
            self.root.bind(face, lambda event, f=face: self.apply_move(f + "'"))

    def apply_move(self, move_name: str) -> None:
        self.cube.apply_move_name(move_name)
        self.move_history.append(move_name)
        self.redraw()

    def reset_cube(self) -> None:
        self.cube = Cube()
        self.move_history.clear()
        self._clear_solution()
        self.redraw()

    def scramble_cube(self) -> None:
        self.cube = Cube()
        self.move_history.clear()
        moves = self.cube.scramble(int(self.scramble_length_var.get()), self.rng)
        self.move_history.extend(MOVE_NAMES[idx] for idx in moves)
        self._clear_solution()
        self.redraw()

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------

    def _clear_solution(self) -> None:
        self._playing = False
        self._solution_moves = []
        self._solution_step = 0
        self._solve_status_var.set("")
        self._step_btn.config(state="disabled")
        self._play_btn.config(state="disabled")

    def _on_solve(self) -> None:
        if self.cube.is_solved():
            self._solve_status_var.set("Already solved!")
            return

        if self._distances is None:
            if self._distances_loading:
                return
            self._distances_loading = True
            self._solve_btn.config(state="disabled")
            self._solve_status_var.set("Loading distance table…")
            threading.Thread(target=self._load_distances, daemon=True).start()
            return

        moves = solve(self.cube.state, self._distances)
        self._solution_moves = moves
        self._solution_step = 0
        self._update_solution_display()
        self._step_btn.config(state="normal")
        self._play_btn.config(state="normal" if moves else "disabled")

    def _load_distances(self) -> None:
        cache = _DISTANCES_CACHE
        try:
            self._distances = DistanceTable.load(cache)
            self.root.after(0, self._on_distances_loaded)
        except FileNotFoundError:
            self.root.after(0, lambda: self._solve_status_var.set(
                "Error: data/distances.npz not found.\nRun: uv run cube-dataset"
            ))
            self.root.after(0, lambda: self._solve_btn.config(state="normal"))
            self._distances_loading = False

    def _on_distances_loaded(self) -> None:
        self._distances_loading = False
        self._solve_btn.config(state="normal")
        self._on_solve()

    def _on_step(self) -> None:
        if self._solution_step >= len(self._solution_moves):
            return
        move_idx = self._solution_moves[self._solution_step]
        self.cube.apply_move(move_idx)
        self.move_history.append(MOVE_NAMES[move_idx])
        self._solution_step += 1
        self.redraw()
        self._update_solution_display()
        if self._solution_step >= len(self._solution_moves):
            self._step_btn.config(state="disabled")
            self._play_btn.config(state="disabled")

    def _on_play(self) -> None:
        if self._playing:
            self._playing = False
            self._play_btn.config(text="▶ Play")
        else:
            if self._solution_step >= len(self._solution_moves):
                return
            self._playing = True
            self._play_btn.config(text="■ Stop")
            self._play_next()

    def _play_next(self) -> None:
        if not self._playing or self._solution_step >= len(self._solution_moves):
            self._playing = False
            self._play_btn.config(text="▶ Play")
            return
        self._on_step()
        if self._solution_step < len(self._solution_moves):
            delay = max(50, int(self._play_delay_var.get()))
            self.root.after(delay, self._play_next)
        else:
            self._playing = False
            self._play_btn.config(text="▶ Play")

    def _update_solution_display(self) -> None:
        if not self._solution_moves:
            self._solve_status_var.set("No solution.")
            return
        names = [MOVE_NAMES[m] for m in self._solution_moves]
        done = names[: self._solution_step]
        remaining = names[self._solution_step :]
        total = len(self._solution_moves)
        left = total - self._solution_step
        parts = []
        if done:
            parts.append(f"[{' '.join(done)}]")
        if remaining:
            parts.append(" ".join(remaining))
        self._solve_status_var.set(f"{left}/{total} moves left: {' '.join(parts)}")

    def redraw(self) -> None:
        self.canvas.delete("all")
        self._draw_cube_net()
        self.cube3d.redraw(self.cube.state)

        solved_faces = int(self.cube.face_solved().sum())
        oriented_corners = int(self.cube.corner_oriented().sum())
        solved_text = "yes" if self.cube.is_solved() else "no"
        self.status_var.set(
            f"Solved: {solved_text}\n"
            f"Faces solved: {solved_faces}/6\n"
            f"Corners oriented: {oriented_corners}/8"
        )

        if self.move_history:
            history = " ".join(self.move_history[-18:])
            prefix = "History" if len(self.move_history) <= 18 else "History (last 18)"
            self.history_var.set(f"{prefix}: {history}")
        else:
            self.history_var.set("History: none")

    def _canvas_dimensions(self) -> tuple[int, int]:
        face_span = 2 * STICKER_SIZE + STICKER_GAP
        cell_span = face_span + FACE_GAP
        max_grid_x = max(grid_x for grid_x, _ in FACE_GRID_POSITIONS.values())
        max_grid_y = max(grid_y for _, grid_y in FACE_GRID_POSITIONS.values())
        width = MARGIN_X * 2 + max_grid_x * cell_span + face_span
        height = MARGIN_Y * 2 + max_grid_y * cell_span + face_span
        return width, height

    def _draw_cube_net(self) -> None:
        for face_name, (grid_x, grid_y) in FACE_GRID_POSITIONS.items():
            face_x = MARGIN_X + grid_x * (2 * STICKER_SIZE + STICKER_GAP + FACE_GAP)
            face_y = MARGIN_Y + grid_y * (2 * STICKER_SIZE + STICKER_GAP + FACE_GAP)
            self._draw_face(face_name, face_x, face_y, STICKER_SIZE, STICKER_GAP)

    def _draw_face(
        self,
        face_name: str,
        x0: int,
        y0: int,
        sticker_size: int,
        gap: int,
    ) -> None:
        self.canvas.create_text(
            x0 + sticker_size,
            y0 - 12,
            text=face_name,
            fill="#e5e7eb",
            font=("Helvetica", 14, "bold"),
        )

        start = FACE_START[face_name]
        face_state = self.cube.state[start : start + 4]

        for row in range(2):
            for col in range(2):
                sticker_idx = row * 2 + col
                color_code = IDX_TO_COLOR[int(face_state[sticker_idx])]
                fill = COLOR_TO_HEX[color_code]
                x1 = x0 + col * (sticker_size + gap)
                y1 = y0 + row * (sticker_size + gap)
                x2 = x1 + sticker_size
                y2 = y1 + sticker_size

                self.canvas.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill=fill,
                    outline="#0f172a",
                    width=2,
                )
                self.canvas.create_text(
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    text=color_code,
                    fill="#111827",
                    font=("Helvetica", 13, "bold"),
                )


def main() -> None:
    root = tk.Tk()
    CubeVisualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
