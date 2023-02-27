"""Dynamically create a configuration for gunicorn from environment variables
"""
import multiprocessing
import os

host = os.getenv("HOST", "0.0.0.0")
port = os.getenv("PORT", "80")
bind_env = os.getenv("BIND", None)
use_loglevel = os.getenv("LOG_LEVEL", "info")
if bind_env:
    use_bind = bind_env
else:
    use_bind = f"{host}:{port}"

# calculate number of workers (can be conrolled via ENV variables)
workers_per_core_str = os.getenv("WORKERS_PER_CORE", "1")
web_concurrency_str = os.getenv("WEB_CONCURRENCY", None)
cores = multiprocessing.cpu_count()
workers_per_core = float(workers_per_core_str)
default_web_concurrency = workers_per_core * cores
if web_concurrency_str:
    web_concurrency = int(web_concurrency_str)
    assert web_concurrency > 0
else:
    web_concurrency = max(int(default_web_concurrency), 2)

# Gunicorn config variables
# See https://docs.gunicorn.org/en/stable/settings.html
loglevel = use_loglevel
workers = web_concurrency
bind = use_bind
keepalive = 120
# errorlog = "-"
# accesslog = "-"
# disable_redirect_access_to_syslog = True
# capture_output = True
# enable_stdio_inheritance = True