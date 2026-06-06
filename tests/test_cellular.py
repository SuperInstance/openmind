"""Tests for the cellular computation layer."""

import json
import os
import time
import pytest

from openmind.cellular import (
    ResourceSnapshot, probe, MetabolicPath, select_path,
    train_or_load, sense_or_simulate, infer_adaptive,
    save_cache, load_cache, cell,
)


class TestResourceSnapshot:
    def test_default_snapshot(self):
        snap = ResourceSnapshot()
        assert snap.gpu_available is False
        assert snap.cpu_cores >= 1

    def test_resource_level_high(self):
        snap = ResourceSnapshot(
            gpu_available=True, network_available=True,
            api_keys={"openai": True}, ram_available_gb=32, cpu_cores=16,
        )
        assert snap.resource_level() == "high"

    def test_resource_level_low(self):
        snap = ResourceSnapshot(gpu_available=False, api_keys={}, network_available=False)
        assert snap.resource_level() == "low"

    def test_resource_level_medium(self):
        snap = ResourceSnapshot(gpu_available=False, api_keys={"openai": True}, network_available=True)
        assert snap.resource_level() in ("medium", "high", "low")

    def test_has_api(self):
        snap = ResourceSnapshot(api_keys={"openai": True, "anthropic": False})
        assert snap.has_api() is True
        assert snap.has_api("openai") is True
        assert snap.has_api("anthropic") is False

    def test_has_gpu(self):
        assert ResourceSnapshot(gpu_available=True).has_gpu() is True
        assert ResourceSnapshot(gpu_available=False).has_gpu() is False

    def test_has_hardware(self):
        assert ResourceSnapshot(esp32_ports=["/dev/ttyUSB0"]).has_hardware() is True
        assert ResourceSnapshot(esp32_ports=[]).has_hardware() is False


class TestProbe:
    def test_probe_returns_snapshot(self):
        snap = probe()
        assert isinstance(snap, ResourceSnapshot)
        assert snap.timestamp > 0
        assert isinstance(snap.api_keys, dict)
        assert isinstance(snap.cpu_cores, int)


class TestMetabolicPath:
    def test_full_train_when_all_resources(self):
        resources = ResourceSnapshot(
            gpu_available=True, gpu_memory_free_mb=8000,
            api_keys={"openai": True}, ram_available_gb=32,
        )
        path = select_path("train", resources)
        assert path == MetabolicPath.FULL_TRAIN

    def test_muscle_memory_when_nothing(self):
        resources = ResourceSnapshot(gpu_available=False, api_keys={}, network_available=False)
        path = select_path("train", resources)
        assert path == MetabolicPath.MUSCLE_MEMORY

    def test_cloud_when_api_only(self):
        resources = ResourceSnapshot(gpu_available=False, api_keys={"openai": True}, network_available=True)
        path = select_path("infer", resources)
        assert path == MetabolicPath.CLOUD_INFERENCE

    def test_hardware_when_esp32(self):
        resources = ResourceSnapshot(esp32_ports=["/dev/ttyUSB0"], gpu_available=False, api_keys={})
        path = select_path("sense", resources)
        assert path == MetabolicPath.HARDWARE_LOOP

    def test_prefer_override(self):
        resources = ResourceSnapshot(gpu_available=True, api_keys={"openai": True})
        assert select_path("x", resources, prefer="cached") == MetabolicPath.MUSCLE_MEMORY
        assert select_path("x", resources, prefer="gpu") == MetabolicPath.FULL_TRAIN


class TestCache:
    def test_save_and_load(self):
        save_cache("test_model", {"weights": [1, 2, 3]})
        loaded = load_cache("test_model")
        assert loaded is not None
        assert loaded["weights"] == [1, 2, 3]

    def test_load_nonexistent(self):
        result = load_cache("nonexistent_xyz_model_12345")
        assert result is None


class TestTrainOrLoad:
    def test_with_train_fn(self):
        def mock_train(name, data):
            return {"type": "trained", "name": name}

        resources = ResourceSnapshot(
            gpu_available=True, api_keys={"openai": True},
        )
        model = train_or_load(
            "test_model", data=[1, 2, 3],
            train_fn=mock_train, resources=resources,
        )
        assert model["type"] == "trained"

    def test_fallback_cached(self):
        resources = ResourceSnapshot(gpu_available=False, api_keys={})
        model = train_or_load("missing_model_999", resources=resources)
        assert model is not None

    def test_fallback_fail(self):
        resources = ResourceSnapshot(gpu_available=False, api_keys={})
        with pytest.raises(RuntimeError):
            train_or_load("missing_model_888", fallback="fail", resources=resources)

    def test_fallback_simulate(self):
        resources = ResourceSnapshot(gpu_available=False, api_keys={})
        model = train_or_load("missing_model_777", fallback="simulate", resources=resources)
        assert model["type"] == "simulated"


class TestSenseOrSimulate:
    def test_with_simulate_fn(self):
        def mock_sim(source, duration, rate, **kw):
            return [0.5, 0.6, 0.7]

        data = sense_or_simulate("temp", simulate_fn=mock_sim)
        assert len(data) == 3

    def test_fallback_synthetic(self):
        data = sense_or_simulate("temp_999")
        assert len(data) > 0


class TestInferAdaptive:
    def test_with_infer_fn(self):
        def mock_infer(model, data):
            return [1, -1, 0]

        result = infer_adaptive({"name": "test"}, [1, 2], infer_fn=mock_infer)
        assert result == [1, -1, 0]

    def test_fallback(self):
        result = infer_adaptive({"name": "missing_999"}, [1, 2], strategy="cached")
        assert result is not None


class TestCellDecorator:
    def test_cell_runs_function(self):
        @cell(resource_aware=True, fallback="cached")
        def my_func():
            return 42

        assert my_func() == 42

    def test_cell_fallback_on_error(self):
        @cell(resource_aware=True, fallback="cached")
        def bad_func():
            raise ValueError("oops")

        result = bad_func()
        assert "error" in result
        assert result["fallback"] == "cached"

    def test_cell_fail_mode(self):
        @cell(resource_aware=True, fallback="fail")
        def bad_func():
            raise ValueError("oops")

        with pytest.raises(ValueError):
            bad_func()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
