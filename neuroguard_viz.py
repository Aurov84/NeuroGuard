"""
NeuroGuard - Geometric SVM-Based Early Warning for Catastrophic Forgetting
Visualization Experiment Platform v1.0

Dependencies: Python 3.x (tkinter built-in) + matplotlib
Run: python neuroguard_viz.py
"""

import math
import copy
import tkinter as tk
from tkinter import ttk
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon
import numpy as np


# ============================================================
# Core Computation Module
# ============================================================


def sigmoid(z):
    z = max(-20, min(20, z))
    return 1 / (1 + math.exp(-z))


def bce_loss(y, y_hat):
    eps = 1e-10
    y_hat = max(eps, min(1 - eps, y_hat))
    return -(y * math.log(y_hat) + (1 - y) * math.log(1 - y_hat))


class Neuron:
    """A neuron with historical memory parameters."""

    def __init__(self, w1=0.5, w2=0.5, bias=0.5):
        self.w1 = w1
        self.w2 = w2
        self.bias = bias
        self.hyperplane_slope = 0
        self.margin_distance = 9999
        self.trained = False
        self.memory_updated = False

    def forward(self, x1, x2):
        return sigmoid(x1 * self.w1 + x2 * self.w2 + self.bias)

    def clone(self):
        return copy.deepcopy(self)

    def copy_from(self, other):
        self.w1 = other.w1
        self.w2 = other.w2
        self.bias = other.bias
        self.hyperplane_slope = other.hyperplane_slope
        self.margin_distance = other.margin_distance
        self.trained = other.trained
        self.memory_updated = other.memory_updated


def train_epochs(neuron, data, epochs, lr=0.1):
    """Train for multiple epochs. Returns (trained neuron copy, loss history)."""
    n = neuron.clone()
    loss_history = []
    for _ in range(epochs):
        total_loss = 0
        for x1, x2, y in data:
            y_hat = n.forward(x1, x2)
            total_loss += bce_loss(y, y_hat)
            error = y_hat - y
            n.w1 -= lr * error * x1
            n.w2 -= lr * error * x2
            n.bias -= lr * error
        loss_history.append(total_loss / len(data))
    n.trained = True
    return n, loss_history


def compute_hyperplane_slope(neuron):
    if abs(neuron.w2) < 1e-12:
        return float("inf")
    return -neuron.w1 / neuron.w2


def compute_margin_distance(neuron, data):
    """Compute distance from hyperplane to nearest data point (half-margin)."""
    w_norm = math.sqrt(neuron.w1**2 + neuron.w2**2)
    if w_norm < 1e-12:
        return 9999
    min_dist = float("inf")
    for x1, x2, y in data:
        dist = abs(neuron.w1 * x1 + neuron.w2 * x2 + neuron.bias) / w_norm
        if dist < min_dist:
            min_dist = dist
    return min_dist


def compute_hyperplane_position(neuron):
    """Compute hyperplane position along the normal direction."""
    w_norm = math.sqrt(neuron.w1**2 + neuron.w2**2)
    if w_norm < 1e-12:
        return 0
    return -neuron.bias / w_norm


def update_memory(neuron, data):
    """Compute and update the neuron's memory parameters."""
    neuron.hyperplane_slope = compute_hyperplane_slope(neuron)
    neuron.margin_distance = compute_margin_distance(neuron, data)
    neuron.memory_updated = True


def compute_similarity(n_old, n_new):
    """Cosine similarity between two normal vectors."""
    a1, a2 = n_old.w1, n_old.w2
    b1, b2 = n_new.w1, n_new.w2
    dot = a1 * b1 + a2 * b2
    norm_a = math.sqrt(a1**2 + a2**2)
    norm_b = math.sqrt(b1**2 + b2**2)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0
    return dot / (norm_a * norm_b)


def compute_margin_overlap(n_old, n_new):
    """Compute whether the margin intervals overlap."""
    old_pos = compute_hyperplane_position(n_old)
    old_m = n_old.margin_distance
    new_pos = compute_hyperplane_position(n_new)
    new_m = n_new.margin_distance
    old_lo, old_hi = old_pos - old_m, old_pos + old_m
    new_lo, new_hi = new_pos - new_m, new_pos + new_m
    overlap_lo = max(old_lo, new_lo)
    overlap_hi = min(old_hi, new_hi)
    overlap_len = max(0, overlap_hi - overlap_lo)
    return {
        "old_pos": old_pos,
        "old_range": (old_lo, old_hi),
        "new_pos": new_pos,
        "new_range": (new_lo, new_hi),
        "overlap_length": overlap_len,
        "has_overlap": overlap_len > 0,
    }


def get_conflict_level(similarity, has_overlap):
    if similarity > 0.9 and has_overlap:
        return "SAFE - Knowledge Compatible", "#2ecc71"
    if similarity > 0.9 and not has_overlap:
        return "WARNING - Positional Conflict", "#f39c12"
    if similarity > 0.5 and has_overlap:
        return "WARNING - Partial Conflict", "#f39c12"
    if similarity > 0.5 and not has_overlap:
        return "DANGER - Significant Conflict", "#e74c3c"
    if similarity <= 0.5 and has_overlap:
        return "WARNING - Large Direction Diff with Overlap", "#f39c12"
    return "CRITICAL - Fundamental Conflict", "#e74c3c"


# ============================================================
# Datasets
# ============================================================

AND_DATA = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
OR_DATA = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]


# ============================================================
# Drawing Functions
# ============================================================

PLOT_RANGE = (-0.5, 1.5)


def get_line_points(w1, w2, bias, x_range):
    """Compute endpoints of line w1*x + w2*y + bias = 0."""
    pts = []
    if abs(w2) > 1e-12:
        for x in x_range:
            y = -(w1 * x + bias) / w2
            pts.append((x, y))
    elif abs(w1) > 1e-12:
        x = -bias / w1
        pts = [(x, x_range[0]), (x, x_range[1])]
    return pts


def draw_neuron_on_ax(
    ax, neuron, data=None, color="#3498db", alpha=1.0, label_prefix=""
):
    """Draw a neuron's hyperplane and margin on a matplotlib axes."""
    ax.set_xlim(PLOT_RANGE)
    ax.set_ylim(PLOT_RANGE)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="#555", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(True, color="#2a2a3e", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("x1", fontsize=8, color="#777")
    ax.set_ylabel("x2", fontsize=8, color="#777")

    if neuron is None:
        return

    w1, w2, bias = neuron.w1, neuron.w2, neuron.bias
    w_norm = math.sqrt(w1**2 + w2**2)
    x_range = (PLOT_RANGE[0] - 1, PLOT_RANGE[1] + 1)

    # Decision boundary
    pts = get_line_points(w1, w2, bias, x_range)
    if len(pts) == 2:
        ax.plot(
            [pts[0][0], pts[1][0]],
            [pts[0][1], pts[1][1]],
            color=color,
            linewidth=2,
            alpha=alpha,
        )

    # Margin region
    if neuron.memory_updated and neuron.margin_distance < 9999 and w_norm > 1e-12:
        margin = neuron.margin_distance
        for sign in [-1, 1]:
            offset_bias = bias + sign * margin * w_norm
            m_pts = get_line_points(w1, w2, offset_bias, x_range)
            if len(m_pts) == 2:
                ax.plot(
                    [m_pts[0][0], m_pts[1][0]],
                    [m_pts[0][1], m_pts[1][1]],
                    color=color,
                    linewidth=1,
                    linestyle="--",
                    alpha=alpha * 0.5,
                )

        pts_upper = get_line_points(w1, w2, bias + margin * w_norm, x_range)
        pts_lower = get_line_points(w1, w2, bias - margin * w_norm, x_range)
        if len(pts_upper) == 2 and len(pts_lower) == 2:
            polygon_pts = [pts_upper[0], pts_upper[1], pts_lower[1], pts_lower[0]]
            poly = Polygon(
                polygon_pts,
                closed=True,
                facecolor=color,
                alpha=alpha * 0.12,
                edgecolor="none",
            )
            ax.add_patch(poly)

    # Normal vector arrow
    if w_norm > 1e-6:
        if abs(w2) > 1e-12:
            mid_x = 0.5
            mid_y = -(w1 * mid_x + bias) / w2
        else:
            mid_y = 0.5
            mid_x = -(w2 * mid_y + bias) / w1
        n_len = 0.25
        nx = w1 / w_norm * n_len
        ny = w2 / w_norm * n_len
        ax.annotate(
            "",
            xy=(mid_x + nx, mid_y + ny),
            xytext=(mid_x, mid_y),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.8, alpha=alpha),
        )
        ax.text(
            mid_x + nx + 0.03,
            mid_y + ny + 0.03,
            "w",
            fontsize=8,
            color=color,
            alpha=alpha,
            fontweight="bold",
        )

    # Data points
    if data is not None:
        for x1, x2, y in data:
            if y == 1:
                ax.plot(
                    x1,
                    x2,
                    "o",
                    color="#e74c3c",
                    markersize=7,
                    markeredgecolor="#ff6b6b",
                    markeredgewidth=1.2,
                    alpha=alpha,
                    zorder=5,
                )
            else:
                ax.plot(
                    x1,
                    x2,
                    "s",
                    color="#9b59b6",
                    markersize=7,
                    markeredgecolor="#bb8fce",
                    markeredgewidth=1.2,
                    alpha=alpha,
                    zorder=5,
                )


# ============================================================
# Main Application
# ============================================================


class NeuroGuardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuroGuard - Catastrophic Forgetting Early Warning Platform")
        self.root.configure(bg="#0d0d1a")

        # Auto-adapt to screen size
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        if screen_w <= 1600 or screen_h <= 900:
            try:
                root.state("zoomed")
            except tk.TclError:
                try:
                    root.attributes("-zoomed", True)
                except tk.TclError:
                    root.geometry(f"{screen_w}x{screen_h}+0+0")
        else:
            win_w, win_h = 1600, 900
            x = (screen_w - win_w) // 2
            y = (screen_h - win_h) // 2
            root.geometry(f"{win_w}x{win_h}+{x}+{y}")

        root.minsize(1200, 700)

        # State
        self.real_neuron = Neuron()
        self.mirror_neuron = None
        self.mirror_data = None
        self.real_data = None
        self.task_name = None
        self.loss_history = []
        self.total_epochs_trained = 0

        # Style
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#0d0d1a")
        self.style.configure(
            "TLabel", background="#0d0d1a", foreground="#aaa", font=("Consolas", 9)
        )
        self.style.configure(
            "Header.TLabel",
            background="#0d0d1a",
            foreground="#ecf0f1",
            font=("Consolas", 14, "bold"),
        )

        self._build_ui()
        self._show_welcome()
        self._refresh_all()

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Header
        header = ttk.Frame(self.root)
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))
        ttk.Label(header, text="NEUROGUARD", style="Header.TLabel").pack(side="left")
        ttk.Label(
            header,
            text="  Geometric SVM-Based Early Warning for Catastrophic Forgetting",
            foreground="#555",
        ).pack(side="left", padx=(5, 0))

        # Main area
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=3)
        main_frame.rowconfigure(1, weight=1)

        # Top: Three panels
        plots_frame = ttk.Frame(main_frame)
        plots_frame.grid(row=0, column=0, sticky="nsew")
        for i in range(3):
            plots_frame.columnconfigure(i, weight=1)
        plots_frame.rowconfigure(1, weight=1)

        titles = [
            ("Real Neuron", "#3498db"),
            ("Comparison View", "#95a5a6"),
            ("Mirror Neuron", "#2ecc71"),
        ]
        for i, (title, color) in enumerate(titles):
            ttk.Label(plots_frame, text=f"  {title}", foreground=color).grid(
                row=0, column=i, sticky="w", padx=12, pady=(4, 0)
            )

        self.fig_main = plt.Figure(facecolor="#0d0d1a", dpi=100)
        self.fig_main.subplots_adjust(
            left=0.04, right=0.98, top=0.95, bottom=0.08, wspace=0.15
        )
        self.ax1 = self.fig_main.add_subplot(131)
        self.ax2 = self.fig_main.add_subplot(132)
        self.ax3 = self.fig_main.add_subplot(133)

        self.canvas_main = FigureCanvasTkAgg(self.fig_main, plots_frame)
        self.canvas_main.get_tk_widget().grid(
            row=1, column=0, columnspan=3, sticky="nsew", padx=4, pady=4
        )

        # Bottom: Loss + Log + Controls
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=1, column=0, sticky="nsew", pady=(2, 0))
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(1, weight=2)
        bottom_frame.columnconfigure(2, weight=0)
        bottom_frame.rowconfigure(0, weight=1)

        # Loss curve
        loss_frame = ttk.Frame(bottom_frame)
        loss_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        ttk.Label(loss_frame, text="Loss Curve", foreground="#e67e22").pack(
            anchor="w", padx=4
        )

        self.fig_loss = plt.Figure(facecolor="#0d0d1a", dpi=100)
        self.fig_loss.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.18)
        self.ax_loss = self.fig_loss.add_subplot(111)
        self.canvas_loss = FigureCanvasTkAgg(self.fig_loss, loss_frame)
        self.canvas_loss.get_tk_widget().pack(fill="both", expand=True, padx=2, pady=2)

        # Log
        log_frame = ttk.Frame(bottom_frame)
        log_frame.grid(row=0, column=1, sticky="nsew", padx=4)
        ttk.Label(log_frame, text="Log", foreground="#95a5a6").pack(anchor="w", padx=4)

        self.log_text = tk.Text(
            log_frame,
            bg="#111122",
            fg="#aaa",
            font=("Consolas", 9),
            relief="flat",
            padx=8,
            pady=6,
            wrap="word",
            state="disabled",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground="#222",
        )
        self.log_text.pack(fill="both", expand=True, padx=2, pady=2)
        self.log_text.tag_configure("info", foreground="#aaa")
        self.log_text.tag_configure("success", foreground="#2ecc71")
        self.log_text.tag_configure("warning", foreground="#f39c12")
        self.log_text.tag_configure("danger", foreground="#e74c3c")
        self.log_text.tag_configure(
            "header", foreground="#3498db", font=("Consolas", 9, "bold")
        )
        self.log_text.tag_configure("param", foreground="#bb8fce")

        # Controls
        ctrl_frame = ttk.Frame(bottom_frame)
        ctrl_frame.grid(row=0, column=2, sticky="nsew", padx=(4, 0))

        input_frame = ttk.Frame(ctrl_frame)
        input_frame.pack(fill="x", padx=4, pady=(0, 4))

        ttk.Label(input_frame, text="Epochs:", foreground="#777").grid(
            row=0, column=0, sticky="w", pady=2
        )
        self.epochs_var = tk.StringVar(value="500")
        tk.Entry(
            input_frame,
            textvariable=self.epochs_var,
            width=8,
            bg="#1a1a2e",
            fg="#ecf0f1",
            font=("Consolas", 9),
            relief="flat",
            insertbackground="#ecf0f1",
            highlightthickness=1,
            highlightbackground="#333",
        ).grid(row=0, column=1, sticky="w", padx=(4, 0), pady=2)

        ttk.Label(input_frame, text="Learning Rate:", foreground="#777").grid(
            row=1, column=0, sticky="w", pady=2
        )
        self.lr_var = tk.StringVar(value="0.1")
        tk.Entry(
            input_frame,
            textvariable=self.lr_var,
            width=8,
            bg="#1a1a2e",
            fg="#ecf0f1",
            font=("Consolas", 9),
            relief="flat",
            insertbackground="#ecf0f1",
            highlightthickness=1,
            highlightbackground="#333",
        ).grid(row=1, column=1, sticky="w", padx=(4, 0), pady=2)

        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.pack(fill="both", expand=True, padx=4)

        self.buttons = {}
        btn_defs = [
            ("clone", "Clone (Create Mirror)", self._on_clone),
            ("task_and", "Task: AND", self._on_task_and),
            ("task_or", "Task: OR", self._on_task_or),
            ("train", "Train", self._on_train),
            ("memory", "Compute Memory", self._on_update_memory),
            ("conflict", "Conflict Detection", self._on_conflict_detect),
            ("writeback", "Write Back to Real", self._on_writeback),
            ("discard", "Discard Mirror", self._on_discard),
        ]
        for key, text, cmd in btn_defs:
            btn = tk.Button(
                btn_frame,
                text=text,
                command=cmd,
                font=("Consolas", 9),
                bg="#1a1a2e",
                fg="#aaa",
                activebackground="#252540",
                activeforeground="#ecf0f1",
                relief="flat",
                anchor="w",
                padx=8,
                pady=3,
                borderwidth=0,
                highlightthickness=1,
                highlightbackground="#2a2a3e",
                cursor="hand2",
            )
            btn.pack(fill="x", pady=1)
            self.buttons[key] = btn

    def _show_welcome(self):
        welcome = """====================================================
     NeuroGuard - Method Overview
====================================================

[Neuron Structure]
  A single neuron has n weights (w) and 1 bias.
  Parameters: w1, w2, bias
  Output: sigmoid(w1*x1 + w2*x2 + bias)

  Weight vector (w1, w2) defines the normal vector
  (direction) of the decision hyperplane.
  Bias determines the hyperplane position.

[Core Idea]
  After training converges, record two quantities:
  1. Hyperplane normal direction
     -> encodes "what was learned"
  2. Half-margin width
     -> encodes "how large the safe region is"

[Experiment Workflow]
  1. Clone  -> copy real neuron to create mirror
  2. Select Task -> AND or OR logic gate
  3. Train  -> mirror learns the new task
  4. Compute Memory -> calculate slope & margin
  5. Conflict Detection -> compare old vs new
  6. Write Back or Discard -> accept or reject

[Why Clone from Real Neuron?]
  The real neuron holds the optimal state.
  Using it as initialization ensures:
  - Hyperplane direction stays consistent
  - Continuity with previously learned knowledge
  Random init could produce a reversed normal
  vector (cosine similarity = -1), which appears
  correct but is geometrically conflicting.

[Early Warning Principle]
  Compare real and mirror neurons:
  1. Normal vector cosine similarity
     -> are directions consistent?
  2. Margin interval intersection
     -> are solution spaces compatible?
  No intersection = conflict
  -> forced training destroys old knowledge
     and may trigger catastrophic forgetting.
====================================================
"""
        self._log(welcome, "header")

    def _log(self, msg, tag="info"):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n", tag)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _get_epochs(self):
        try:
            val = int(self.epochs_var.get())
            if val < 1:
                val = 1
        except (ValueError, TypeError):
            val = 1
        self.epochs_var.set(str(val))
        return val

    def _get_lr(self):
        try:
            val = float(self.lr_var.get())
            if val <= 0 or val > 10:
                val = 0.1
        except (ValueError, TypeError):
            val = 0.1
        self.lr_var.set(str(val))
        return val

    def _update_button_states(self):
        has_mirror = self.mirror_neuron is not None
        has_task = self.task_name is not None
        mirror_trained = has_mirror and self.mirror_neuron.trained
        mirror_memory = has_mirror and self.mirror_neuron.memory_updated
        real_memory = self.real_neuron.memory_updated

        states = {
            "clone": not has_mirror,
            "task_and": has_mirror and not mirror_trained,
            "task_or": has_mirror and not mirror_trained,
            "train": has_mirror and has_task,
            "memory": mirror_trained,
            "conflict": real_memory and mirror_memory,
            "writeback": mirror_memory,
            "discard": has_mirror,
        }
        for key, enabled in states.items():
            btn = self.buttons[key]
            if enabled:
                btn.configure(
                    state="normal",
                    fg="#ccc",
                    bg="#1a1a2e",
                    cursor="hand2",
                    highlightbackground="#3498db",
                )
            else:
                btn.configure(
                    state="disabled",
                    fg="#333",
                    bg="#111122",
                    cursor="arrow",
                    highlightbackground="#1a1a2e",
                )

    # --- Action Callbacks ---

    def _on_clone(self):
        self.mirror_neuron = self.real_neuron.clone()
        # Reset mirror state - it needs new task and training
        self.mirror_neuron.trained = False
        self.mirror_neuron.memory_updated = False
        self.mirror_data = None
        self.task_name = None
        self.loss_history = []
        self.total_epochs_trained = 0
        self._log("Cloned real neuron -> mirror created", "success")
        self._log(
            f"  Mirror: w1={self.mirror_neuron.w1:.4f}, w2={self.mirror_neuron.w2:.4f}, bias={self.mirror_neuron.bias:.4f}",
            "param",
        )
        self._refresh_all()

    def _on_task_and(self):
        self.mirror_data = AND_DATA
        self.task_name = "AND"
        self._log("Task selected: AND", "info")
        self._log("  Data: (0,0)->0  (0,1)->0  (1,0)->0  (1,1)->1", "param")
        self._refresh_all()

    def _on_task_or(self):
        self.mirror_data = OR_DATA
        self.task_name = "OR"
        self._log("Task selected: OR", "info")
        self._log("  Data: (0,0)->0  (0,1)->1  (1,0)->1  (1,1)->1", "param")
        self._refresh_all()

    def _on_train(self):
        if not self.mirror_neuron or not self.mirror_data:
            return
        epochs = self._get_epochs()
        lr = self._get_lr()
        self._log(f"Training... ({epochs} epochs, lr={lr})", "info")

        result, loss_hist = train_epochs(
            self.mirror_neuron, self.mirror_data, epochs, lr
        )
        self.mirror_neuron = result
        self.loss_history.extend(loss_hist)
        self.total_epochs_trained += epochs

        final_loss = loss_hist[-1]
        self._log(
            f"Training complete! Total {self.total_epochs_trained} epochs", "success"
        )
        self._log(f"  Final loss = {final_loss:.6f}", "param")
        self._log("  Predictions:", "info")

        correct = 0
        for x1, x2, y in self.mirror_data:
            y_hat = self.mirror_neuron.forward(x1, x2)
            pred = 1 if y_hat >= 0.5 else 0
            if pred == y:
                correct += 1
            self._log(
                f"    ({x1},{x2}) expected={y} predicted={y_hat:.4f} {'Y' if pred == y else 'N'}",
                "success" if pred == y else "danger",
            )

        acc = correct / len(self.mirror_data) * 100
        self._log(f"  Accuracy: {acc:.0f}%", "success" if acc == 100 else "warning")
        self._log(
            f"  Params: w1={result.w1:.4f}, w2={result.w2:.4f}, bias={result.bias:.4f}",
            "param",
        )
        self._refresh_all()

    def _on_update_memory(self):
        if not self.mirror_neuron or not self.mirror_data:
            return
        update_memory(self.mirror_neuron, self.mirror_data)
        slope = self.mirror_neuron.hyperplane_slope
        margin = self.mirror_neuron.margin_distance
        self._log("Memory parameters updated (mirror)", "success")
        self._log(
            (
                f"  Hyperplane slope = {slope:.6f}"
                if slope != float("inf")
                else "  Hyperplane slope = inf"
            ),
            "param",
        )
        self._log(f"  Half-margin = {margin:.6f}", "param")
        self._refresh_all()

    def _on_conflict_detect(self):
        if (
            not self.mirror_neuron
            or not self.real_neuron.memory_updated
            or not self.mirror_neuron.memory_updated
        ):
            return
        similarity = compute_similarity(self.real_neuron, self.mirror_neuron)
        overlap = compute_margin_overlap(self.real_neuron, self.mirror_neuron)
        level_text, _ = get_conflict_level(similarity, overlap["has_overlap"])

        self._log("======= Conflict Detection Report =======", "header")
        self._log(
            f"  Result: {level_text}",
            (
                "success"
                if "SAFE" in level_text
                else ("warning" if "WARNING" in level_text else "danger")
            ),
        )
        self._log(f"  Cosine similarity: {similarity:.6f}", "param")
        self._log(
            f"  Real margin: [{overlap['old_range'][0]:.4f}, {overlap['old_range'][1]:.4f}]",
            "param",
        )
        self._log(
            f"  Mirror margin: [{overlap['new_range'][0]:.4f}, {overlap['new_range'][1]:.4f}]",
            "param",
        )
        self._log(
            f"  Intersection: {'Yes' if overlap['has_overlap'] else 'No'} (overlap={overlap['overlap_length']:.4f})",
            "success" if overlap["has_overlap"] else "danger",
        )
        self._log("=========================================", "header")
        self._refresh_all()

    def _on_writeback(self):
        if not self.mirror_neuron:
            return
        self.real_neuron.copy_from(self.mirror_neuron)
        self.real_data = self.mirror_data
        self._log("Mirror written back to real neuron!", "success")
        self._log(
            f"  Real: w1={self.real_neuron.w1:.4f}, w2={self.real_neuron.w2:.4f}, bias={self.real_neuron.bias:.4f}",
            "param",
        )
        self.mirror_neuron = None
        self.mirror_data = None
        self.task_name = None
        self.loss_history = []
        self.total_epochs_trained = 0
        self._refresh_all()

    def _on_discard(self):
        self.mirror_neuron = None
        self.mirror_data = None
        self.task_name = None
        self.loss_history = []
        self.total_epochs_trained = 0
        self._log("Mirror discarded", "warning")
        self._refresh_all()

    # --- Refresh Views ---

    def _refresh_all(self):
        self._update_button_states()
        self._draw_panel1()
        self._draw_panel2()
        self._draw_panel3()
        self._draw_loss()
        self.canvas_main.draw_idle()
        self.canvas_loss.draw_idle()

    def _draw_panel1(self):
        """Panel 1: Real neuron (always visible, no data points)"""
        self.ax1.clear()
        draw_neuron_on_ax(
            self.ax1, self.real_neuron, data=None, color="#3498db", alpha=1.0
        )
        self.ax1.set_title("Real Neuron", fontsize=9, color="#3498db", pad=4)
        n = self.real_neuron
        self.ax1.text(
            0.02,
            0.02,
            f"w1={n.w1:.3f}  w2={n.w2:.3f}  bias={n.bias:.3f}",
            transform=self.ax1.transAxes,
            fontsize=7,
            color="#3498db",
            alpha=0.7,
            verticalalignment="bottom",
            fontfamily="monospace",
        )
        if n.memory_updated:
            slope_s = (
                f"{n.hyperplane_slope:.3f}"
                if n.hyperplane_slope != float("inf")
                else "inf"
            )
            self.ax1.text(
                0.02,
                0.08,
                f"slope={slope_s}  half-margin={n.margin_distance:.3f}",
                transform=self.ax1.transAxes,
                fontsize=7,
                color="#3498db",
                alpha=0.6,
                verticalalignment="bottom",
                fontfamily="monospace",
            )

    def _draw_panel2(self):
        """Panel 2: Comparison - real (base) + mirror (overlay)"""
        self.ax2.clear()
        draw_neuron_on_ax(
            self.ax2, self.real_neuron, data=None, color="#3498db", alpha=0.6
        )
        if self.mirror_neuron is not None:
            draw_neuron_on_ax(
                self.ax2,
                self.mirror_neuron,
                data=self.mirror_data,
                color="#2ecc71",
                alpha=0.5,
            )
        self.ax2.set_title("Comparison View", fontsize=9, color="#95a5a6", pad=4)
        self.ax2.text(
            0.02,
            0.02,
            "Blue=Real  Green=Mirror",
            transform=self.ax2.transAxes,
            fontsize=7,
            color="#888",
            verticalalignment="bottom",
            fontfamily="monospace",
        )

    def _draw_panel3(self):
        """Panel 3: Mirror neuron + task data"""
        self.ax3.clear()
        if self.mirror_neuron is not None:
            draw_neuron_on_ax(
                self.ax3,
                self.mirror_neuron,
                data=self.mirror_data,
                color="#2ecc71",
                alpha=1.0,
            )
            self.ax3.set_title("Mirror Neuron", fontsize=9, color="#2ecc71", pad=4)
            if self.task_name:
                self.ax3.text(
                    0.98,
                    0.98,
                    f"Task: {self.task_name}",
                    transform=self.ax3.transAxes,
                    fontsize=8,
                    color="#2ecc71",
                    alpha=0.7,
                    va="top",
                    ha="right",
                    fontfamily="monospace",
                )
            n = self.mirror_neuron
            self.ax3.text(
                0.02,
                0.02,
                f"w1={n.w1:.3f}  w2={n.w2:.3f}  bias={n.bias:.3f}",
                transform=self.ax3.transAxes,
                fontsize=7,
                color="#2ecc71",
                alpha=0.7,
                verticalalignment="bottom",
                fontfamily="monospace",
            )
            if n.memory_updated:
                slope_s = (
                    f"{n.hyperplane_slope:.3f}"
                    if n.hyperplane_slope != float("inf")
                    else "inf"
                )
                self.ax3.text(
                    0.02,
                    0.08,
                    f"slope={slope_s}  half-margin={n.margin_distance:.3f}",
                    transform=self.ax3.transAxes,
                    fontsize=7,
                    color="#2ecc71",
                    alpha=0.6,
                    verticalalignment="bottom",
                    fontfamily="monospace",
                )
        else:
            self.ax3.set_facecolor("#0d0d1a")
            self.ax3.set_xlim(0, 1)
            self.ax3.set_ylim(0, 1)
            self.ax3.text(
                0.5,
                0.5,
                "Waiting for clone...",
                transform=self.ax3.transAxes,
                fontsize=10,
                color="#333",
                ha="center",
                va="center",
                fontfamily="monospace",
            )
            self.ax3.set_xticks([])
            self.ax3.set_yticks([])
            for spine in self.ax3.spines.values():
                spine.set_color("#1a1a2e")
            self.ax3.set_title("Mirror Neuron", fontsize=9, color="#333", pad=4)

    def _draw_loss(self):
        self.ax_loss.clear()
        self.ax_loss.set_facecolor("#1a1a2e")
        self.ax_loss.tick_params(colors="#555", labelsize=7)
        for spine in self.ax_loss.spines.values():
            spine.set_color("#333")
        self.ax_loss.grid(True, color="#2a2a3e", linewidth=0.5, alpha=0.5)
        self.ax_loss.set_xlabel("Epoch", fontsize=8, color="#777")
        self.ax_loss.set_ylabel("Loss", fontsize=8, color="#777")
        self.ax_loss.set_title(
            "Training Loss Curve", fontsize=9, color="#e67e22", pad=4
        )
        if self.loss_history:
            self.ax_loss.plot(
                range(1, len(self.loss_history) + 1),
                self.loss_history,
                color="#e67e22",
                linewidth=1.5,
                alpha=0.9,
            )
            self.ax_loss.text(
                0.98,
                0.95,
                f"Final loss = {self.loss_history[-1]:.6f}",
                transform=self.ax_loss.transAxes,
                fontsize=7,
                color="#e67e22",
                ha="right",
                va="top",
                fontfamily="monospace",
            )
        else:
            self.ax_loss.text(
                0.5,
                0.5,
                "No training data yet",
                transform=self.ax_loss.transAxes,
                fontsize=9,
                color="#333",
                ha="center",
                va="center",
                fontfamily="monospace",
            )


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuroGuardApp(root)
    root.mainloop()
