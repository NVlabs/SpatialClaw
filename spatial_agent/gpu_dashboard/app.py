"""FastAPI app for the GPU dashboard (server-centric / job-wise)."""

import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote as _url_quote

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import BaseLoader, Environment

from . import templates
from .agent_counter import snapshot as agent_snapshot
from .discovery import Server, build_servers, summarize_servers
from .storage import GpuDashboardDB


class _StringLoader(BaseLoader):
    TEMPLATES = {
        "base.html": templates.BASE_LAYOUT,
        "overview.html": templates.OVERVIEW_PAGE,
        "history.html": templates.HISTORY_PAGE,
        "server_detail.html": templates.SERVER_DETAIL_PAGE,
        "agents.html": templates.AGENTS_PAGE,
    }

    def get_source(self, environment, name):
        src = self.TEMPLATES.get(name)
        if src is None:
            raise Exception(f"Template not found: {name}")
        return src, name, lambda: True


_jinja = Environment(loader=_StringLoader(), autoescape=True)
_jinja.globals["urlencode"] = lambda s: _url_quote(str(s), safe="")
_jinja.filters["urlencode"] = lambda s: _url_quote(str(s), safe="")


def _util_class(v: Optional[float]) -> str:
    if v is None:
        return "warn"
    if v >= 80:
        return "bad"
    if v >= 50:
        return "warn"
    return "good"


def create_app(db_path: str, project_root: Optional[Path] = None) -> FastAPI:
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent
    project_root = Path(project_root)
    db = GpuDashboardDB(db_path)
    app = FastAPI(title="Spatial Agent GPU Dashboard")

    def _render(template: str, **ctx) -> HTMLResponse:
        tmpl = _jinja.get_template(template)
        ctx.setdefault("page", template.replace(".html", ""))
        return HTMLResponse(tmpl.render(**ctx))

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    # ------------------------------------------------------------------ HTML

    def _build_overview() -> dict:
        servers = build_servers(project_root)
        latest = db.latest_samples()
        snap = agent_snapshot(project_root)

        by_server: dict[str, list[dict]] = {}
        last_ts_by_server: dict[str, int] = {}
        for r in latest:
            sid = r.get("service_id") or ""
            by_server.setdefault(sid, []).append(r)
            if r["ts"] > last_ts_by_server.get(sid, 0):
                last_ts_by_server[sid] = r["ts"]

        now = int(time.time())
        cards: list[dict] = []
        for srv in servers:
            gpus = []
            for r in sorted(
                by_server.get(srv.server_id, []), key=lambda r: r["gpu_index"]
            ):
                gpus.append({
                    "gpu_index": r["gpu_index"],
                    "util_pct": r.get("util_pct"),
                    "mem_used_mb": r.get("mem_used_mb"),
                    "mem_total_mb": r.get("mem_total_mb"),
                    "temp_c": r.get("temp_c"),
                    "power_w": r.get("power_w"),
                    "mem_label": _fmt_mem(
                        r.get("mem_used_mb"), r.get("mem_total_mb")
                    ),
                    "temp_label": _fmt_temp(r.get("temp_c")),
                    "power_label": _fmt_power(r.get("power_w")),
                })
            last_ts = last_ts_by_server.get(srv.server_id, 0)
            cards.append({
                "server_id": srv.server_id,
                "service_type": srv.service_type,
                "display_label": srv.display_label,
                "model": srv.model,
                "tools": srv.tools,
                "node": srv.node,
                "ip": srv.ip,
                "slurm_job_id": srv.slurm_job_id,
                "pid": srv.pid,
                "num_gpus": srv.num_gpus,
                "gpus": gpus,
                "gpus_expected": srv.gpus_hint,
                "last_ts": last_ts,
                "stale": (now - last_ts) > 60 if last_ts else True,
                "last_ago": "—" if not last_ts else _fmt_ago(now - last_ts),
            })

        cards.sort(
            key=lambda c: (c["service_type"], c["display_label"], c["server_id"])
        )

        gpu_count = sum(len(c["gpus"]) for c in cards)
        utils = [
            g["util_pct"]
            for c in cards for g in c["gpus"]
            if g["util_pct"] is not None
        ]
        mean_util = round(sum(utils) / len(utils)) if utils else 0
        peak_util = round(max(utils)) if utils else 0
        peak_label = ""
        if utils:
            peak_entry = max(
                ((g, c) for c in cards for g in c["gpus"] if g["util_pct"] is not None),
                key=lambda x: x[0]["util_pct"],
            )
            peak_label = (
                f"{peak_entry[1]['display_label']} "
                f"GPU {peak_entry[0]['gpu_index']}"
            )

        distinct_nodes = {c["node"] for c in cards if c["node"]}
        summary = {
            "server_count": len(cards),
            "node_count": len(distinct_nodes),
            "gpu_count": gpu_count,
            "agent_total": snap.total,
            "agent_breakdown": snap.by_benchmark,
            "mean_util": mean_util,
            "mean_util_class": _util_class(mean_util),
            "peak_util": peak_util,
            "peak_util_class": _util_class(peak_util),
            "peak_label": peak_label or "—",
            "last_ts": max(last_ts_by_server.values(), default=0),
        }
        return {"cards": cards, "summary": summary, "now": now}

    @app.get("/", response_class=HTMLResponse)
    def overview():
        data = _build_overview()
        return _render(
            "overview.html",
            cards=data["cards"], summary=data["summary"], page="overview",
        )

    @app.get("/api/overview")
    def api_overview():
        return JSONResponse(_build_overview())

    @app.get("/history", response_class=HTMLResponse)
    def history_page():
        servers = build_servers(project_root)
        active = [(s.server_id, s.display_label, s.service_type) for s in servers]
        historic = db.distinct_servers(max_age_sec=30 * 86400)
        have = {sid for sid, _, _ in active}
        for row in historic:
            sid = row["service_id"]
            if sid and sid not in have:
                active.append((sid, sid, row.get("service_type") or ""))
                have.add(sid)
        active.sort(key=lambda t: (t[2], t[1], t[0]))
        return _render("history.html", servers=active, page="history")

    @app.get("/servers/{server_id:path}", response_class=HTMLResponse)
    def server_page(server_id: str):
        servers = {s.server_id: s for s in build_servers(project_root)}
        srv = servers.get(server_id)
        if srv is None:
            raise HTTPException(status_code=404, detail="Server not found")
        return _render(
            "server_detail.html",
            server=_server_context(srv),
            page="server",
        )

    @app.get("/agents", response_class=HTMLResponse)
    def agents_page():
        snap = agent_snapshot(project_root)
        snap_dict = {
            "total": snap.total,
            "by_benchmark": snap.by_benchmark,
            "by_model": snap.by_model,
            "experiment_ids": snap.experiment_ids,
        }
        return _render("agents.html", snap=snap_dict, page="agents")

    # ------------------------------------------------------------------ API

    @app.get("/api/current")
    def api_current():
        rows = db.latest_samples()
        servers = summarize_servers(build_servers(project_root))
        snap = agent_snapshot(project_root)
        return JSONResponse({
            "latest": rows,
            "servers": servers,
            "agents": {
                "total": snap.total,
                "by_benchmark": snap.by_benchmark,
                "by_model": snap.by_model,
                "experiment_ids": snap.experiment_ids,
            },
        })

    @app.get("/api/history")
    def api_history(
        since: int = Query(...),
        until: Optional[int] = None,
        bucket: int = 300,
        node: Optional[str] = None,
        service: Optional[str] = None,
        server_id: Optional[str] = None,
    ):
        rows = db.gpu_history(
            since=since, until=until, node=node,
            service_type=service, service_id=server_id,
            bucket_sec=bucket,
        )
        return JSONResponse({"rows": rows, "bucket": bucket})

    @app.get("/api/servers/{server_id:path}/history")
    def api_server_history(
        server_id: str,
        since: int = Query(...),
        until: Optional[int] = None,
        bucket: int = 300,
    ):
        rows = db.gpu_history_per_gpu_server(
            since=since, until=until, service_id=server_id, bucket_sec=bucket,
        )
        return JSONResponse({"rows": rows, "bucket": bucket})

    @app.get("/api/agents/history")
    def api_agents_history(
        since: int = Query(...),
        until: Optional[int] = None,
        bucket: int = 300,
    ):
        rows = db.agent_history(since=since, until=until, bucket_sec=bucket)
        return JSONResponse({"rows": rows, "bucket": bucket})

    @app.get("/api/agents/breakdown")
    def api_agents_breakdown(
        since: int = Query(...),
        until: Optional[int] = None,
    ):
        rows = db.agent_breakdown_series(since=since, until=until)
        return JSONResponse({"rows": rows})

    return app


def _server_context(srv: Server) -> dict:
    return {
        "server_id": srv.server_id,
        "service_type": srv.service_type,
        "display_label": srv.display_label,
        "model": srv.model,
        "tools": srv.tools,
        "node": srv.node,
        "ip": srv.ip,
        "slurm_job_id": srv.slurm_job_id,
        "pid": srv.pid,
        "num_gpus": srv.num_gpus,
        "gpus_expected": srv.gpus_hint,
    }


def _fmt_mem(used_mb, total_mb) -> str:
    if used_mb is None or total_mb is None:
        return "— mem"
    g = 1024.0
    return f"{used_mb/g:.1f}/{total_mb/g:.0f} GiB"


def _fmt_temp(c) -> str:
    return "—°C" if c is None else f"{int(c)}°C"


def _fmt_power(w) -> str:
    return "— W" if w is None else f"{int(w)} W"


def _fmt_ago(sec: int) -> str:
    if sec < 60:
        return f"{sec}s ago"
    if sec < 3600:
        return f"{sec // 60}m ago"
    if sec < 86400:
        return f"{sec // 3600}h ago"
    return f"{sec // 86400}d ago"
