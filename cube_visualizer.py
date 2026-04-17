"""Interactive tkinter visualizer for the 2x2x2 cube simulator."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np # pyright: ignore[reportMissingImports]

from cube import Cube, IDX_TO_COLOR, MOVE_NAMES


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

        self._build_ui()
        self._bind_shortcuts()
        self.redraw()

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
        self.redraw()

    def scramble_cube(self) -> None:
        self.cube = Cube()
        self.move_history.clear()
        moves = self.cube.scramble(int(self.scramble_length_var.get()), self.rng)
        self.move_history.extend(MOVE_NAMES[idx] for idx in moves)
        self.redraw()

    def redraw(self) -> None:
        self.canvas.delete("all")
        self._draw_cube_net()

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
