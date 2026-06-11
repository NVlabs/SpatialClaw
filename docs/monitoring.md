# Monitoring & Logs

Dashboards, log locations, per-sample outputs, and how to stop services.

> ⬅ [Main README](../README.md) &nbsp;·&nbsp; Prev: [« Running experiments](running.md) &nbsp;·&nbsp; Next: [Configuration »](configuration.md)

---

## Dashboards

```bash
python -m spatial_agent.launch_managers.vllm_manager        # [1] vLLM dashboard
python -m spatial_agent.launch_managers.gpu_server_manager  # [1] GPU server dashboard
python -m spatial_agent.launch_managers.agent_manager       # [1] Agent dashboard
```

## SLURM logs

```bash
tail -f spatial_agent/logs/slurm_vllm/vllm-<served_name>_<job_id>.out
tail -f spatial_agent/logs/slurm_gpu_server/gpu-<chain>_<job_id>.out
tail -f spatial_agent/logs/slurm_agent/spatial-<benchmark>_<job_id>.out
tail -f spatial_agent/logs/slurm_agent/chain_<benchmark>_<exp>_<id>.log
```

## Per-sample outputs

A run writes to `spatial_agent/work_dir/<work_dir_name>/`:

```
work_dir/<run>/
├── config.json                         # full resolved config for the run
├── predictions.jsonl                   # one JSON prediction per sample
├── results_summary.json                # aggregate accuracy + per-question-type breakdown
├── report_pdf/<sample_id>.pdf          # rendered per-sample PDF report
└── session-<sample_id>/                # one directory per sample
    ├── session_report.html             #   interactive HTML trace
    ├── trace.jsonl                     #   raw event log (init, code cells, tool calls, …)
    ├── show_images/                    #   every image registered via show(...)
    └── vlm_queries/                    #   isolated vlm.locate / vlm.ask_with_thinking transcripts
```

`session_report.html` is the most useful artifact for debugging — it shows every code cell, its stdout, every variable created, every `show()` image, and the final answer.

---

## Stopping Services

### Via the managers

```bash
python -m spatial_agent.launch_managers.vllm_manager        # [3] Stop Server
python -m spatial_agent.launch_managers.gpu_server_manager  # [3] Stop GPU Server(s)
python -m spatial_agent.launch_managers.agent_manager       # [4] Stop Experiment(s)
```

### Manual

```bash
scancel -u $USER                                     # cancel all SLURM jobs
pkill -f "spatial_agent.launch_managers"             # kill background chain managers
```

---

> Next: [Configuration »](configuration.md)
