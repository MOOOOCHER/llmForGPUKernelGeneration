import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kernelgym.config import settings
from kernelgym.server.task_manager import TaskManager


class FakeRedis:
    def __init__(self, tasks=None):
        self.tasks = tasks or {}

    async def hgetall(self, key):
        return self.tasks.get(key, {})


class TaskArchiveTest(unittest.IsolatedAsyncioTestCase):
    async def test_submission_archives_exact_kernel_and_redacted_request(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            manager = TaskManager(FakeRedis())
            task_data = {
                "task_id": "sample_kernel",
                "base_task_id": "sample",
                "kernel_code": "def kernel():\n    return 1",
                "reference_code": "def reference():\n    return 1\n",
                "backend": "triton",
            }

            with patch.object(settings, "task_archive_dir", temporary_dir):
                await manager._archive_submission("sample_kernel", task_data)

            archive_dir = Path(temporary_dir) / "sample"
            self.assertEqual(
                (archive_dir / "kernel.py").read_text(),
                task_data["kernel_code"],
            )
            request = json.loads((archive_dir / "request.json").read_text())
            self.assertEqual(request["task_id"], "sample_kernel")
            self.assertNotIn("kernel_code", request)
            self.assertNotIn("reference_code", request)

    async def test_kernel_result_uses_base_task_directory(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            task_data = {
                "task_id": "sample_kernel",
                "base_task_id": "sample",
                "kernel_code": "pass",
            }
            redis = FakeRedis(
                {
                    "kernelgym:task:sample_kernel": {
                        b"data": json.dumps(task_data).encode(),
                    }
                }
            )
            manager = TaskManager(redis)

            with patch.object(settings, "task_archive_dir", temporary_dir):
                await manager._archive_result(
                    "sample_kernel",
                    "completed",
                    "completed_at",
                    "2026-07-21T10:00:00",
                    {"result": {"compiled": True}},
                )

            result = json.loads(
                (Path(temporary_dir) / "sample" / "kernel_result.json").read_text()
            )
            self.assertEqual(result["task_id"], "sample_kernel")
            self.assertTrue(result["result"]["compiled"])

    async def test_reference_result_is_not_archived(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            manager = TaskManager(FakeRedis())

            with patch.object(settings, "task_archive_dir", temporary_dir):
                await manager._archive_result(
                    "sample_ref",
                    "completed",
                    "completed_at",
                    "2026-07-21T10:00:00",
                    {"result": {}},
                )

            self.assertEqual(list(Path(temporary_dir).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
