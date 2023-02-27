#!/usr/bin/env python
import os
from logger import init_logging
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from api.examples.router import router as examples_router
from api.system.router import router as system_router


# The app
app = FastAPI(title="fastapi-try-out", root_path="", docs_url="/")

# Initialise logger
init_logging()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="https?:\\/\\/(localhost|127\\.0\\.0\\.1)(:\\d{1,5})?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(elastic_router, prefix="/elastic")
app.include_router(system_router, tags=["System"])