import os

import pytest

import grass.script as gs


@pytest.fixture(scope="module")
def grass_session_env(tmp_path_factory):
    """Mirror a GRASS session into os.environ for the module's tests

    Most of the I_cluster_*() functions call G_debug() internally, which
    fatal-errors if GISRC is not set in the process environment, even
    though clustering itself never reads or writes a project or mapset.
    Since these tests call the C functions directly through ctypes rather
    than through grass.script, the session has to be visible in the real
    process environment rather than passed as an explicit env= argument,
    so it is mirrored the same way python/grass/temporal's tests do for
    grass.temporal (see AGENTS.md and grass_temporal_gui_support_test.py).
    """
    project = tmp_path_factory.mktemp("data") / "project"
    gs.create_project(project)
    with (
        pytest.MonkeyPatch.context() as monkeypatch,
        gs.setup.init(project, env=os.environ.copy()) as session,
    ):
        for key, value in session.env.items():
            if os.environ.get(key) != value:
                monkeypatch.setenv(key, value)
        yield session
