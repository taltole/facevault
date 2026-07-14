"""
FaceVault Utils — Tkinter Staff Dashboard UI.

StaffUI runs in Thread 6 and shows:
  - Live RGB camera feed with face annotations
  - Toggle checkboxes: RGB / Depth / Blend / MediaPipe overlay
  - Customer identity badge panel
  - VIP alert banner (flashes gold)
  - Zone dwell progress bars
  - Brightness/contrast sliders

Based on actual utils_tk.py by Tal Toledano.
"""

import tkinter as tk
from tkinter import ttk, font as tkfont
import cv2
import numpy as np
import logging
from PIL import Image, ImageTk
from threading import Lock

logger = logging.getLogger("facevault.ui")


class StaffUI:
    """
    Tkinter-based staff monitoring dashboard.

    Usage (from Thread 6):
        ui = StaffUI("FaceVault")
        while running:
            ui.update_frame(annotated_frame)
            ui.update_metadata({"customer_id": "cust_001", ...})
            ui.update()   # process Tk events
    """

    WINDOW_W = 1000
    WINDOW_H = 680

    def __init__(self, title: str = "FaceVault — Staff Dashboard"):
        self._lock       = Lock()
        self._vip_flash  = 0

        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{self.WINDOW_W}x{self.WINDOW_H}")
        self.root.configure(bg="#1a1a2e")
        self.root.attributes("-topmost", False)

        self._build_ui()
        logger.info("StaffUI initialised")

    def _build_ui(self):
        """Build the two-panel layout: left=camera, right=info."""
        # ── Left: Camera feed ─────────────────────────────────────────────────
        left = tk.Frame(self.root, bg="#1a1a2e")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # VIP alert banner (hidden by default)
        self.vip_banner = tk.Label(
            left, text="", bg="#FFD700", fg="#000000",
            font=("Helvetica", 14, "bold"), height=2
        )
        self.vip_banner.pack(fill=tk.X, pady=(0, 4))

        # Camera display
        self.display_label = tk.Label(left, bg="#0d0d0d")
        self.display_label.pack(fill=tk.BOTH, expand=True)

        # ── Right: Controls + Info ────────────────────────────────────────────
        right = tk.Frame(self.root, bg="#16213e", width=260)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)
        right.pack_propagate(False)

        # Title
        tk.Label(right, text="FaceVault", bg="#16213e", fg="#e94560",
                 font=("Helvetica", 16, "bold")).pack(pady=(10, 2))
        tk.Label(right, text="Staff Monitor", bg="#16213e", fg="#a0a0c0",
                 font=("Helvetica", 10)).pack(pady=(0, 12))

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, padx=8, pady=4)

        # ── Frame toggles ─────────────────────────────────────────────────────
        tk.Label(right, text="DISPLAY", bg="#16213e", fg="#a0a0c0",
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=12, pady=(8,2))

        self.show_rgb    = tk.BooleanVar(value=True)
        self.show_depth  = tk.BooleanVar(value=False)
        self.show_blend  = tk.BooleanVar(value=False)
        self.show_mp     = tk.BooleanVar(value=True)
        self.flip_ud     = tk.BooleanVar(value=False)
        self.flip_lr     = tk.BooleanVar(value=False)

        for text, var in [
            ("RGB Feed",        self.show_rgb),
            ("Depth Map",       self.show_depth),
            ("RGB + Depth",     self.show_blend),
            ("MediaPipe Mesh",  self.show_mp),
            ("Flip Vertical",   self.flip_ud),
            ("Flip Horizontal", self.flip_lr),
        ]:
            tk.Checkbutton(
                right, text=text, variable=var,
                bg="#16213e", fg="#c0c0e0", selectcolor="#0f3460",
                activebackground="#16213e", font=("Helvetica", 10)
            ).pack(anchor="w", padx=12)

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, padx=8, pady=8)

        # ── Brightness slider ─────────────────────────────────────────────────
        tk.Label(right, text="BRIGHTNESS", bg="#16213e", fg="#a0a0c0",
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=12)
        self.brightness = tk.DoubleVar(value=1.0)
        tk.Scale(right, variable=self.brightness, from_=0.3, to=2.0,
                 resolution=0.05, orient=tk.HORIZONTAL, bg="#16213e", fg="#c0c0e0",
                 highlightthickness=0, length=200).pack(padx=12)

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, padx=8, pady=8)

        # ── Customer info panel ───────────────────────────────────────────────
        tk.Label(right, text="LAST DETECTION", bg="#16213e", fg="#a0a0c0",
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=12)

        self.id_label   = tk.Label(right, text="—", bg="#16213e", fg="#e0e0ff",
                                   font=("Helvetica", 12, "bold"))
        self.id_label.pack(anchor="w", padx=12, pady=2)

        self.conf_label = tk.Label(right, text="Confidence: —", bg="#16213e",
                                   fg="#80c0ff", font=("Helvetica", 10))
        self.conf_label.pack(anchor="w", padx=12)

        self.zone_label = tk.Label(right, text="Zone: —", bg="#16213e",
                                   fg="#80ffc0", font=("Helvetica", 10))
        self.zone_label.pack(anchor="w", padx=12)

        self.dwell_label = tk.Label(right, text="Dwell: —", bg="#16213e",
                                    fg="#ffd080", font=("Helvetica", 10))
        self.dwell_label.pack(anchor="w", padx=12)

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, padx=8, pady=8)

        # ── Visit counter ─────────────────────────────────────────────────────
        tk.Label(right, text="TODAY", bg="#16213e", fg="#a0a0c0",
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=12)

        self.total_label    = tk.Label(right, text="Visits: 0", bg="#16213e",
                                       fg="#e0e0ff", font=("Helvetica", 11))
        self.total_label.pack(anchor="w", padx=12)

        self.returning_label = tk.Label(right, text="Returning: 0", bg="#16213e",
                                        fg="#80ffc0", font=("Helvetica", 11))
        self.returning_label.pack(anchor="w", padx=12)

        self.vip_count_label = tk.Label(right, text="VIP: 0 ★", bg="#16213e",
                                         fg="#FFD700", font=("Helvetica", 11, "bold"))
        self.vip_count_label.pack(anchor="w", padx=12)

    # ── Public API ────────────────────────────────────────────────────────────

    def update_frame(self, frame: np.ndarray) -> None:
        """Push a new BGR frame to the camera display label."""
        if frame is None:
            return

        # Apply display transforms
        if self.flip_ud.get():
            frame = cv2.flip(frame, 0)
        if self.flip_lr.get():
            frame = cv2.flip(frame, 1)

        alpha = self.brightness.get()
        if alpha != 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)

        # Fit frame to display area
        h, w    = frame.shape[:2]
        max_w   = self.WINDOW_W - 280
        max_h   = self.WINDOW_H - 60
        scale   = min(max_w / w, max_h / h)
        new_w   = int(w * scale)
        new_h   = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        tk_img  = ImageTk.PhotoImage(image=pil_img)

        self.display_label.config(image=tk_img)
        self.display_label._image = tk_img   # keep reference

    def update_metadata(self, metadata: dict) -> None:
        """Update the info panel with latest detection metadata."""
        cid  = metadata.get("customer_id", "—")
        conf = metadata.get("confidence",   0.0)
        zone = metadata.get("zone",         "—")
        dwl  = metadata.get("dwell_sec",    0.0)
        tot  = metadata.get("total_today",  0)
        ret  = metadata.get("returning",    0)
        vip  = metadata.get("vip_count",    0)

        prefix = "★ " if metadata.get("is_vip") else ""
        self.id_label.config(text=f"{prefix}{cid}")
        self.conf_label.config(text=f"Confidence: {conf:.0%}")
        self.zone_label.config(text=f"Zone: {zone}")
        self.dwell_label.config(text=f"Dwell: {dwl:.1f}s")
        self.total_label.config(text=f"Visits: {tot}")
        self.returning_label.config(text=f"Returning: {ret}")
        self.vip_count_label.config(text=f"VIP: {vip} ★")

    def show_vip_alert(self, customer_id: str) -> None:
        """Flash a gold VIP alert banner for 3 seconds."""
        self.vip_banner.config(text=f"★  VIP CUSTOMER DETECTED: {customer_id}  ★")
        self._vip_flash = 90   # frames to show banner (~3s at 30fps)

    def update(self) -> None:
        """Process Tkinter event queue — call once per frame."""
        if self._vip_flash > 0:
            self._vip_flash -= 1
            if self._vip_flash % 10 < 5:
                self.vip_banner.config(bg="#FFD700")
            else:
                self.vip_banner.config(bg="#B8860B")
        else:
            self.vip_banner.config(text="", bg="#1a1a2e")

        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            pass

    def destroy(self) -> None:
        """Clean shutdown of the Tk window."""
        try:
            self.root.destroy()
        except Exception:
            pass
