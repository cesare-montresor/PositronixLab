"""SQLite layout.  One experiment directory:

  ax.sqlite               Ax's own tables, all phases (one Ax experiment per phase).
                          Ax keeps ONE engine per process, so one file per process.
  <phase>.results.sqlite  trials, all metrics, events, meta (written here), one per phase

Only the main process writes.  Trial processes never touch the databases.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class ExperimentPaths:
    def __init__(self, root: str | Path, name: str):
        self.dir = Path(root) / name

    def ax_db(self) -> Path:
        """ONE Ax database per experiment: Ax's SQL store has a process-global engine."""
        return self.dir / "ax.sqlite"

    def results_db(self, phase: str) -> Path:
        return self.dir / f"{phase}.results.sqlite"

    def ax_url(self) -> str:
        return f"sqlite:///{self.ax_db().resolve()}"

    def phases(self) -> list[str]:
        return sorted(p.name[: -len(".results.sqlite")] for p in self.dir.glob("*.results.sqlite"))

    def ensure(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    def delete(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


_AX_LOCK = threading.Lock()
_AX_URL: str | None = None


def init_ax_db(path: str | Path) -> str:
    """Initialise Ax's process-global SQLAlchemy engine on ``path`` and create its tables.

    Idempotent for the same existing file; re-initialises when the path changed or the file
    was deleted (restart).  One experiment per process at a time.
    """
    global _AX_URL
    from ax.storage.sqa_store import db

    path = Path(path)
    url = f"sqlite:///{path.resolve()}"
    with _AX_LOCK:
        if db.SESSION_FACTORY is not None and _AX_URL == url and path.exists():
            return url
        path.parent.mkdir(parents=True, exist_ok=True)
        # busy timeout: the BO phases (threads) write the same file
        db.init_engine_and_session_factory(url=url, force_init=True, connect_args={"timeout": 60})
        db.create_all_tables(db.get_engine())
        with sqlite3.connect(path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
        _AX_URL = url
    return url


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=str)


class ResultsStore:
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
    CREATE TABLE IF NOT EXISTS trials(
        trial_index INTEGER PRIMARY KEY, source TEXT NOT NULL, origin INTEGER,
        status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 1,
        params TEXT NOT NULL, config TEXT, error TEXT, created REAL, finished REAL);
    CREATE TABLE IF NOT EXISTS metrics(
        trial_index INTEGER NOT NULL, name TEXT NOT NULL, value REAL,
        step INTEGER NOT NULL DEFAULT -1, PRIMARY KEY(trial_index, name, step));
    CREATE TABLE IF NOT EXISTS events(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, trial_index INTEGER,
        kind TEXT, detail TEXT);
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        with self._lock, self._conn:
            self._conn.executescript(self._SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # meta ---------------------------------------------------------------
    def set_meta(self, key: str, value: Any) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, _dumps(value)))

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    # trials -------------------------------------------------------------
    def add_trial(self, trial_index: int, source: str, params: dict, config: dict | None = None,
                  status: str = "running", origin: int | None = None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO trials(trial_index, source, origin, status, params, config, created)"
                " VALUES (?,?,?,?,?,?,?)",
                (trial_index, source, origin, status, _dumps(params),
                 _dumps(config) if config is not None else None, time.time()))

    def set_status(self, trial_index: int, status: str, error: str | None = None) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE trials SET status=?, error=?, finished=? WHERE trial_index=?",
                               (status, error, time.time(), trial_index))

    def bump_attempt(self, trial_index: int) -> int:
        with self._lock, self._conn:
            self._conn.execute("UPDATE trials SET attempts = attempts + 1 WHERE trial_index=?", (trial_index,))
            return self._conn.execute("SELECT attempts FROM trials WHERE trial_index=?",
                                      (trial_index,)).fetchone()[0]

    def add_metrics(self, trial_index: int, metrics: dict[str, float], step: int = -1) -> None:
        rows = [(trial_index, k, None if v is None else float(v), step) for k, v in metrics.items()]
        with self._lock, self._conn:
            self._conn.executemany("INSERT OR REPLACE INTO metrics VALUES (?,?,?,?)", rows)

    def add_event(self, trial_index: int | None, kind: str, detail: Any = None) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT INTO events(ts, trial_index, kind, detail) VALUES (?,?,?,?)",
                               (time.time(), trial_index, kind, _dumps(detail)))

    def trials(self, status: str | None = None, source: str | None = None) -> list[dict]:
        q, args = "SELECT trial_index, source, origin, status, attempts, params, config, error FROM trials", []
        cond = []
        if status:
            cond.append("status=?"); args.append(status)
        if source:
            cond.append("source=?"); args.append(source)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        with self._lock:
            rows = self._conn.execute(q + " ORDER BY trial_index", args).fetchall()
            mrows = self._conn.execute("SELECT trial_index, name, value FROM metrics WHERE step=-1").fetchall()
        metrics: dict[int, dict] = {}
        for idx, name, value in mrows:
            metrics.setdefault(idx, {})[name] = value
        return [{"trial_index": r[0], "source": r[1], "origin": r[2], "status": r[3], "attempts": r[4],
                 "params": json.loads(r[5]), "config": json.loads(r[6]) if r[6] else None,
                 "error": r[7], "metrics": metrics.get(r[0], {})} for r in rows]

    def count(self, source: str | None = None) -> int:
        q, args = "SELECT COUNT(*) FROM trials", []
        if source:
            q += " WHERE source=?"; args.append(source)
        with self._lock:
            return self._conn.execute(q, args).fetchone()[0]

    def events(self, kind: str | None = None) -> list[dict]:
        q, args = "SELECT ts, trial_index, kind, detail FROM events", []
        if kind:
            q += " WHERE kind=?"; args.append(kind)
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [{"ts": r[0], "trial_index": r[1], "kind": r[2], "detail": json.loads(r[3])} for r in rows]
