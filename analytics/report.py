"""
FaceVault — Analytics & Reporting
===================================
Generates weekly customer behaviour reports from the HDF5 visit database.

Uses:
  utils_file  — reads HDF5 visit log, exports JSON report
  utils_vis   — t-SNE customer cluster scatter, zone heatmap
  utils_gl    — 3D attention heatmap on store floor plan
"""

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

from Utils.utils_file import VisitLogger, NumpyEncoder
from Utils.utils_vis  import show_embedds, draw_gaze_heatmap
from Utils.utils_gl   import render_attention_heatmap_3d

from config.config import DB_PATH, REPORT_DIR, ZONE_MAP

logger = logging.getLogger("facevault.analytics")


class WeeklyReporter:
    """
    Reads 7 days of visit records from HDF5 and produces:
      1. JSON summary (total_visits, unique_customers, avg_dwell, zone_breakdown)
      2. t-SNE scatter of customer face embeddings (cluster = customer segment)
      3. Zone attention heatmap (2D matplotlib)
      4. 3D OpenGL attention heatmap on store floor plan (utils_gl)
    """

    def __init__(self, db_path=DB_PATH, report_dir=REPORT_DIR):
        self.visit_logger = VisitLogger(db_path)
        self.report_dir   = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def load_week(self, days_back=7):
        """Load all visits from the past N days."""
        cutoff = datetime.now() - timedelta(days=days_back)
        visits = self.visit_logger.load_visits_since(cutoff)
        logger.info(f"Loaded {len(visits)} visits since {cutoff.date()}")
        return visits

    def compute_summary(self, visits):
        """Aggregate visit stats into a summary dict."""
        if not visits:
            return {}

        total_visits      = len(visits)
        unique_customers  = len({v["customer_id"] for v in visits})
        returning         = len({v["customer_id"] for v in visits if v.get("is_returning")})
        vip_visits        = sum(1 for v in visits if v.get("is_vip"))

        dwell_times = [v.get("dwell_sec", 0) for v in visits]
        avg_dwell   = float(np.mean(dwell_times)) if dwell_times else 0.0
        max_dwell   = float(np.max(dwell_times))  if dwell_times else 0.0

        # Zone breakdown — how many attention-seconds per zone
        zone_dwell = defaultdict(float)
        for v in visits:
            for zone, secs in v.get("zone_dwell", {}).items():
                zone_dwell[zone] += secs

        # Peak hour analysis
        hours = [datetime.fromisoformat(v["ts_in"]).hour for v in visits if "ts_in" in v]
        peak_hour = int(np.bincount(hours).argmax()) if hours else None

        summary = {
            "generated_at":      datetime.now().isoformat(),
            "period_days":       7,
            "total_visits":      total_visits,
            "unique_customers":  unique_customers,
            "returning_rate":    round(returning / max(unique_customers, 1), 3),
            "vip_visits":        vip_visits,
            "avg_dwell_sec":     round(avg_dwell, 1),
            "max_dwell_sec":     round(max_dwell, 1),
            "zone_attention_sec": dict(zone_dwell),
            "peak_hour":         peak_hour,
            "top_zone":          max(zone_dwell, key=zone_dwell.get) if zone_dwell else None,
        }
        return summary

    def plot_customer_segments(self, visits, out_path=None):
        """
        Uses utils_vis.show_embedds to run t-SNE on face embeddings
        and scatter-plot customer segments. Each point = one unique customer.
        Colour = number of return visits.
        """
        embeddings  = []
        visit_counts = []

        customer_embs = defaultdict(list)
        for v in visits:
            if "embedding" in v:
                customer_embs[v["customer_id"]].append(v["embedding"])

        for cid, embs in customer_embs.items():
            # Average embedding across visits = stable identity representation
            avg_emb = np.mean(embs, axis=0)
            embeddings.append(avg_emb)
            visit_counts.append(len(embs))

        if len(embeddings) < 5:
            logger.warning("Not enough customers for t-SNE (need >= 5)")
            return

        embeddings   = np.array(embeddings)
        visit_counts = np.array(visit_counts)

        # t-SNE via utils_vis
        show_embedds(
            embeddings=embeddings,
            labels=visit_counts,
            title="Customer Segments — Face Embedding Space\n(colour = visit frequency)",
            out_path=out_path or self.report_dir / "customer_segments.png"
        )
        logger.info("Saved customer segment plot")

    def plot_zone_heatmap(self, visits, out_path=None):
        """
        2D heatmap of customer attention per store zone.
        Uses utils_vis.draw_gaze_heatmap.
        """
        # Build per-zone dwell time array
        zone_names  = list(ZONE_MAP.keys())
        zone_totals = []

        for zone in zone_names:
            total = sum(v.get("zone_dwell", {}).get(zone, 0) for v in visits)
            zone_totals.append(total)

        draw_gaze_heatmap(
            zone_names=zone_names,
            zone_values=zone_totals,
            title="Zone Attention Heatmap — Total Dwell (seconds)",
            out_path=out_path or self.report_dir / "zone_heatmap.png"
        )
        logger.info("Saved zone heatmap")

    def render_3d_heatmap(self, visits):
        """
        3D OpenGL render of customer attention on a store floor plan.
        Uses utils_gl.render_attention_heatmap_3d — spawns a GLUT window
        showing the floor plan with height-mapped attention peaks per zone.
        """
        zone_data = {}
        for zone_name, bbox in ZONE_MAP.items():
            total_dwell = sum(v.get("zone_dwell", {}).get(zone_name, 0) for v in visits)
            zone_data[zone_name] = {
                "bbox":  bbox,
                "dwell": total_dwell,
            }

        render_attention_heatmap_3d(
            zone_data=zone_data,
            title="FaceVault — 3D Attention Heatmap"
        )

    def export_json(self, summary, out_path=None):
        """Exports summary dict to JSON using NumpyEncoder (handles numpy types)."""
        path = out_path or self.report_dir / f"weekly_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(path, "w") as f:
            json.dump(summary, f, cls=NumpyEncoder, indent=2)
        logger.info(f"Report saved to {path}")
        return path

    def run(self):
        """Full report generation pipeline."""
        logger.info("Starting weekly report generation")

        visits  = self.load_week(days_back=7)
        summary = self.compute_summary(visits)

        logger.info(f"Summary: {summary['total_visits']} visits, "
                    f"{summary['unique_customers']} unique customers, "
                    f"top zone: {summary.get('top_zone')}")

        self.plot_customer_segments(visits)
        self.plot_zone_heatmap(visits)
        json_path = self.export_json(summary)

        print("\n" + "=" * 50)
        print("  WEEKLY REPORT SUMMARY")
        print("=" * 50)
        print(f"  Total visits:      {summary['total_visits']}")
        print(f"  Unique customers:  {summary['unique_customers']}")
        print(f"  Returning rate:    {summary['returning_rate']*100:.1f}%")
        print(f"  Avg dwell time:    {summary['avg_dwell_sec']}s")
        print(f"  Peak hour:         {summary.get('peak_hour')}:00")
        print(f"  Top attention zone:{summary.get('top_zone')}")
        print(f"  VIP visits:        {summary['vip_visits']}")
        print("=" * 50)
        print(f"  Report saved: {json_path}")
        print("=" * 50 + "\n")

        return summary


if __name__ == "__main__":
    reporter = WeeklyReporter()
    reporter.run()
