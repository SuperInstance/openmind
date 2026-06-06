"""Tests for openmind Jupyter magic (import only, no IPython required)."""

import pytest


def test_magic_import():
    """Test that magic module can be imported."""
    # IPython may not be installed, so we catch ImportError
    try:
        from openmind.jupyter.magic import OpenMindMagics
        assert OpenMindMagics is not None
    except ImportError:
        pytest.skip("IPython not installed")


def test_dashboard_import():
    """Test that dashboard module can be imported."""
    from openmind.jupyter.dashboard import render_analysis, render_search, render_decision, render_hardware
    assert render_analysis is not None


def test_dashboard_render_analysis():
    """Test rendering analysis HTML."""
    from openmind.jupyter.dashboard import render_analysis
    from openmind import IngestResult, Decision

    result = IngestResult(
        repo_url="test",
        local_path="/tmp",
        functions=[],
        classes=[],
        test_files=[],
        file_structure={},
        call_graph={},
    )
    html = render_analysis(result, {})
    assert "<html" not in html  # It's an HTML fragment, not a full document
    assert "OpenMind" in html
    assert isinstance(html, str)


def test_dashboard_render_decision():
    """Test rendering decision HTML."""
    from openmind.jupyter.dashboard import render_decision
    from openmind import Decision

    html = render_decision("test_func", Decision.HYBRID)
    assert "HYBRID" in html
    assert isinstance(html, str)


def test_dashboard_render_hardware():
    """Test rendering hardware HTML."""
    from openmind.jupyter.dashboard import render_hardware
    from openmind import HardwareCapabilities

    cap = HardwareCapabilities(gpu=True, gpu_name="Test GPU", ram_gb=16.0, cpu_cores=8)
    html = render_hardware(cap)
    assert "16.0 GB" in html
    assert "Test GPU" in html
