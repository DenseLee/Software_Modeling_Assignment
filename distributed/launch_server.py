"""
distributed/launch_server.py — Start the parameter server with a live ngrok tunnel.

Usage (on the host machine)
---------------------------
    cd "D:\\VSCODE\\PyhtonStuff\\AI Project"
    python -m distributed.launch_server

Options
-------
    --port          Local port to bind (default: 8000)
    --domain        Static ngrok domain (overrides config)
    --auth-token    ngrok auth token (or set NGROK_AUTHTOKEN env var)
    --checkpoint    Path to a checkpoint .pt file to resume from
    --no-ngrok      Skip ngrok and serve on LAN only (prints local IP)

Workers on other devices connect to the printed public URL.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

log = logging.getLogger("dist.launch_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Market-Sim distributed parameter server")
    parser.add_argument("--port",       type=int, default=8000)
    parser.add_argument("--domain",     type=str, default=None,
                        help="Static ngrok domain (e.g. free-xxx.ngrok-free.app)")
    parser.add_argument("--auth-token", type=str, default=None,
                        help="ngrok auth token (or set NGROK_AUTHTOKEN env var)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from a .pt checkpoint file")
    parser.add_argument("--no-ngrok",  action="store_true",
                        help="Skip ngrok; serve on LAN only")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from distributed.config import TrainConfig
    from distributed.server import build_app

    cfg = TrainConfig(server_port=args.port)
    if args.domain:
        cfg.ngrok_domain = args.domain

    # ── Build the FastAPI app + learner ───────────────────────────────────────
    app, learner = build_app(cfg)

    # ── Optionally resume from checkpoint ─────────────────────────────────────
    if args.checkpoint:
        import torch
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        learner.model.load_state_dict(ckpt["model_state"])
        learner.opt.load_state_dict(ckpt["opt_state"])
        learner.policy_version = ckpt["policy_version"]
        learner.total_updates  = ckpt["total_updates"]
        log.info(f"Resumed from {args.checkpoint} — policy v={learner.policy_version}")

    # ── Start ngrok tunnel ────────────────────────────────────────────────────
    public_url = None

    if not args.no_ngrok:
        try:
            from pyngrok import ngrok, conf

            auth_token = args.auth_token or os.environ.get("NGROK_AUTHTOKEN")
            if auth_token:
                conf.get_default().auth_token = auth_token

            connect_kwargs = {"addr": args.port, "proto": "http"}
            if cfg.ngrok_domain:
                connect_kwargs["domain"] = cfg.ngrok_domain

            tunnel = ngrok.connect(**connect_kwargs)
            public_url = tunnel.public_url

            print()
            print("=" * 60)
            print(f"  Parameter Server PUBLIC URL:")
            print(f"  {public_url}")
            print(f"  API Docs : {public_url}/docs")
            print(f"  Status   : {public_url}/status")
            print()
            print(f"  Run this on every worker device:")
            print(f"  python -m distributed.launch_worker \\")
            print(f"      --server {public_url} --workers 2")
            print("=" * 60)
            print()

        except ImportError:
            log.warning("pyngrok not installed — run:  pip install pyngrok")
            log.warning("Falling back to LAN mode.")
            args.no_ngrok = True
        except Exception as exc:
            log.error(f"ngrok failed: {exc}")
            log.warning("Falling back to LAN mode.")
            args.no_ngrok = True

    if args.no_ngrok or public_url is None:
        import socket
        local_ip = socket.gethostbyname(socket.gethostname())
        print()
        print("=" * 60)
        print(f"  Parameter Server (LAN only):")
        print(f"  http://{local_ip}:{args.port}")
        print(f"  http://127.0.0.1:{args.port}/docs")
        print()
        print(f"  Workers on same LAN:")
        print(f"  python -m distributed.launch_worker --server http://{local_ip}:{args.port}")
        print("=" * 60)
        print()

    # ── Start uvicorn ─────────────────────────────────────────────────────────
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
