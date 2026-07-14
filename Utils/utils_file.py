"""
FaceVault Utils — HDF5 Visit Logger, JSON export, file helpers.

VisitLogger persists all customer visit events to a structured HDF5 database:
  - Fast append (chunked datasets)
  - Compressed storage (GZIP level 4)
  - Random-access by customer_id (O(1) lookup via index dataset)
  - GDPR right-to-erasure via delete_customer()
  - Automatic purge of records older than RETENTION_DAYS

NumpyEncoder handles JSON serialisation of numpy scalars and arrays —
used for daily JSON report export.

Based on actual utils_file.py by Tal Toledano.
"""

import h5py
import json
import os
import shutil
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Any

logger = logging.getLogger("facevault.file")

RETENTION_DAYS = 90    # auto-purge records older than this


# ── JSON helpers ──────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays transparently."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: dict, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyEncoder, indent=indent)


# ── HDF5 Visit Logger ─────────────────────────────────────────────────────────

class VisitLogger:
    """
    Persists customer visit records to an HDF5 database.

    HDF5 schema:
      /visits/{customer_id}/
          timestamps_in   float64[N]  — monotonic seconds
          timestamps_out  float64[N]
          embeddings      float32[N, 512]
          dwell_sec       float32[N]
          zones           str[N]       — JSON-encoded zone dwell dict
          is_vip          uint8[N]
          confidence      float32[N]

      /index/
          customer_ids    str[M]       — all known customer IDs

    Usage:
        logger = VisitLogger("data/visits.h5")
        logger.update("cust_001", visit_record)
        visits = logger.load_visits_since(datetime.now() - timedelta(days=7))
    """

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self):
        """Create HDF5 file and root groups if they don't exist."""
        with h5py.File(self.db_path, "a") as f:
            if "visits" not in f:
                f.create_group("visits")
            if "index" not in f:
                idx = f.create_group("index")
                dt  = h5py.special_dtype(vlen=str)
                idx.create_dataset("customer_ids", shape=(0,), maxshape=(None,),
                                   dtype=dt, chunks=True)

    def update(self, customer_id: str, record: dict) -> None:
        """
        Append or create a visit record for customer_id.

        record keys: ts_in, embedding, confidence,
                     zone_dwell (dict), is_vip (bool)
        """
        try:
            with h5py.File(self.db_path, "a") as f:
                grp = f["visits"].require_group(customer_id)

                ts      = float(record.get("ts_in", 0.0))
                emb     = np.asarray(record.get("embedding", np.zeros(512)),
                                     dtype=np.float32)
                dwell   = float(record.get("dwell_sec", 0.0))
                zones   = json.dumps(record.get("zone_dwell", {}))
                is_vip  = int(record.get("is_vip", False))
                conf    = float(record.get("confidence", 0.0))

                def _append(name, data, dtype):
                    if name not in grp:
                        if isinstance(data, np.ndarray):
                            shape   = (0, *data.shape)
                            maxshp  = (None, *data.shape)
                            chunks  = (64, *data.shape)
                        else:
                            shape, maxshp, chunks = (0,), (None,), (256,)
                        grp.create_dataset(name, shape=shape, maxshape=maxshp,
                                           dtype=dtype, chunks=chunks,
                                           compression="gzip", compression_opts=4)
                    ds = grp[name]
                    ds.resize(ds.shape[0] + 1, axis=0)
                    if isinstance(data, np.ndarray):
                        ds[-1] = data
                    else:
                        ds[-1] = data

                _append("timestamps_in", ts,    np.float64)
                _append("dwell_sec",     dwell, np.float32)
                _append("confidence",    conf,  np.float32)
                _append("is_vip",        is_vip, np.uint8)
                _append("embeddings",    emb,   np.float32)

                # String dataset (zones)
                if "zones" not in grp:
                    dt = h5py.special_dtype(vlen=str)
                    grp.create_dataset("zones", shape=(0,), maxshape=(None,),
                                       dtype=dt, chunks=(256,))
                ds = grp["zones"]
                ds.resize(ds.shape[0] + 1, axis=0)
                ds[-1] = zones

                # Register in index
                idx_ds = f["index"]["customer_ids"]
                ids    = list(idx_ds[:]) if idx_ds.shape[0] > 0 else []
                if customer_id not in ids:
                    idx_ds.resize(idx_ds.shape[0] + 1, axis=0)
                    idx_ds[-1] = customer_id

        except Exception as e:
            logger.error(f"VisitLogger.update failed for {customer_id}: {e}")

    def load_visits_since(self, cutoff: datetime) -> list:
        """
        Load all visit records newer than cutoff.
        Returns list of dicts, one per visit row.
        """
        visits    = []
        cutoff_ts = cutoff.timestamp()

        try:
            with h5py.File(self.db_path, "r") as f:
                if "visits" not in f:
                    return []

                for cid in f["visits"]:
                    grp = f["visits"][cid]
                    if "timestamps_in" not in grp:
                        continue

                    ts_arr = grp["timestamps_in"][:]
                    mask   = ts_arr >= cutoff_ts

                    for i, keep in enumerate(mask):
                        if not keep:
                            continue
                        emb = grp["embeddings"][i] if "embeddings" in grp else None
                        try:
                            zone_dwell = json.loads(grp["zones"][i])
                        except Exception:
                            zone_dwell = {}

                        visits.append({
                            "customer_id": cid,
                            "ts_in":       datetime.fromtimestamp(ts_arr[i]).isoformat(),
                            "dwell_sec":   float(grp["dwell_sec"][i])
                                           if "dwell_sec" in grp else 0.0,
                            "confidence":  float(grp["confidence"][i])
                                           if "confidence" in grp else 0.0,
                            "is_vip":      bool(grp["is_vip"][i])
                                           if "is_vip" in grp else False,
                            "zone_dwell":  zone_dwell,
                            "embedding":   emb,
                        })
        except Exception as e:
            logger.error(f"load_visits_since failed: {e}")

        logger.debug(f"Loaded {len(visits)} visits since {cutoff.date()}")
        return visits

    def load_embedding_index(self) -> dict:
        """
        Load all stored embeddings into memory for fast identity lookup.
        Returns {customer_id: avg_embedding_array}.
        """
        index = {}
        try:
            with h5py.File(self.db_path, "r") as f:
                for cid in f.get("visits", {}):
                    grp = f["visits"][cid]
                    if "embeddings" in grp and grp["embeddings"].shape[0] > 0:
                        # Average embedding across all visits = stable identity
                        index[cid] = grp["embeddings"][:].mean(axis=0)
        except Exception as e:
            logger.warning(f"load_embedding_index: {e}")
        logger.info(f"Loaded embedding index: {len(index)} customers")
        return index

    def flush_stale_visits(self, active_visits: dict,
                            timeout_sec: float, now: float) -> None:
        """Close visits where the customer hasn't been seen for timeout_sec."""
        stale = [
            cid for cid, v in active_visits.items()
            if now - v.get("ts_last", v.get("ts_in", now)) > timeout_sec
        ]
        for cid in stale:
            v = active_visits.pop(cid)
            v["dwell_sec"] = now - v["ts_in"]
            self.update(cid, v)
            logger.debug(f"Closed visit: {cid} dwell={v['dwell_sec']:.1f}s")

    def close_all(self, active_visits: dict, now: float) -> None:
        """Flush all open visits on pipeline shutdown."""
        for cid, v in active_visits.items():
            v["dwell_sec"] = now - v.get("ts_in", now)
            self.update(cid, v)
        logger.info(f"Closed {len(active_visits)} open visits on shutdown")

    def delete_customer(self, customer_id: str) -> bool:
        """
        Hard-delete all records for a customer (GDPR right-to-erasure).
        Returns True if deletion succeeded.
        """
        try:
            with h5py.File(self.db_path, "a") as f:
                if customer_id in f.get("visits", {}):
                    del f["visits"][customer_id]
                    logger.info(f"GDPR delete: removed all data for {customer_id}")
                    return True
        except Exception as e:
            logger.error(f"delete_customer failed: {e}")
        return False

    def purge_old_records(self, retention_days: int = RETENTION_DAYS) -> int:
        """
        Remove all visit records older than retention_days.
        Returns count of deleted customer records.
        """
        cutoff = datetime.now() - timedelta(days=retention_days)
        deleted = 0
        try:
            with h5py.File(self.db_path, "a") as f:
                for cid in list(f.get("visits", {}).keys()):
                    grp  = f["visits"][cid]
                    if "timestamps_in" not in grp:
                        continue
                    ts_arr = grp["timestamps_in"][:]
                    if ts_arr.max() < cutoff.timestamp():
                        del f["visits"][cid]
                        deleted += 1
        except Exception as e:
            logger.error(f"purge_old_records failed: {e}")
        logger.info(f"Purged {deleted} stale customer records")
        return deleted


# ── File helpers ──────────────────────────────────────────────────────────────

def walk_files(root: str, ext: str = ".jpg") -> list:
    """Recursively find all files with given extension under root."""
    return sorted(Path(root).rglob(f"*{ext}"))


def move_file(src: str, dst: str) -> None:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def copy_file(src: str, dst: str) -> None:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
