"""Start the API on all network interfaces and print the address Laptop 2 should use.

    python -m api [--port 8000]
"""
import argparse
import socket

import uvicorn


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))   # no packet is sent; only selects the outgoing interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    ip = lan_ip()
    print(f"\n  PathSnap API\n  this laptop:   http://127.0.0.1:{a.port}\n  from Laptop 2: http://{ip}:{a.port}"
          f"\n  interactive docs: http://{ip}:{a.port}/docs\n")
    uvicorn.run("api.main:app", host="0.0.0.0", port=a.port, log_level="warning")
