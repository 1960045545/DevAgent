import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from core.todo import Task, TodoList, TodoToolset, is_complex_task


class TaskGraphTests(unittest.TestCase):
    def test_create_exposes_canonical_task_fields_and_dag(self) -> None:
        todo = TodoList()
        snapshot = todo.create(
            [
                {"task_id": "research", "task": "Research", "summary": "scope"},
                {"task_id": "build", "task": "Build", "dependencies": ["research"]},
                {"task": "Verify", "dependencies": ["build"]},
            ]
        )

        self.assertIsInstance(todo.get("research"), Task)
        self.assertEqual(snapshot["edges"], [
            {"from": "research", "to": "build"},
            {"from": "build", "to": "todo-3"},
        ])
        self.assertEqual(snapshot["graph"]["topological_order"], [
            "research",
            "build",
            "todo-3",
        ])
        self.assertEqual(snapshot["tasks"][0]["task"], "Research")
        self.assertEqual(snapshot["tasks"][0]["summary"], "scope")
        self.assertEqual(snapshot["ready_task_ids"], ["research"])

    def test_explicit_actions_enforce_claim_and_complete_protocol(self) -> None:
        todo = TodoList()
        todo.create(["Implement"])

        with self.assertRaisesRegex(ValueError, "in_process"):
            todo.complete("todo-1", "done")
        with self.assertRaisesRegex(ValueError, "pending to completed"):
            todo.update("todo-1", "completed")

        todo.claim("todo-1")
        self.assertEqual(todo.get("todo-1").status, "in_process")
        todo.complete("todo-1", "implemented and tested")
        self.assertEqual(todo.get("todo-1").status, "completed")
        self.assertEqual(todo.get("todo-1").summary, "implemented and tested")

        with self.assertRaisesRegex(ValueError, "pending"):
            todo.claim("todo-1")

    def test_block_propagates_to_pending_dependents(self) -> None:
        todo = TodoList()
        todo.create(
            [
                "Fetch source",
                {"task": "Parse source", "dependencies": ["todo-1"]},
                {"task": "Report", "dependencies": ["todo-2"]},
            ]
        )

        todo.block("todo-1", "source is unavailable")
        self.assertEqual(
            [task.status for task in todo.tasks],
            ["blocked", "blocked", "blocked"],
        )
        self.assertEqual(todo.snapshot()["blocked_task_ids"], [
            "todo-1",
            "todo-2",
            "todo-3",
        ])
        self.assertIn("dependency todo-2 is blocked", todo.get("todo-3").blocked_reason)
        self.assertTrue(todo.is_terminal)

    def test_claim_rejects_a_task_with_unfinished_dependencies(self) -> None:
        todo = TodoList()
        todo.create([
            "Fetch source",
            {"task": "Parse source", "dependencies": ["todo-1"]},
        ])

        with self.assertRaisesRegex(ValueError, "waiting for dependency todo-1"):
            todo.claim("todo-2")
        with self.assertRaisesRegex(ValueError, "not ready"):
            todo.update("todo-2", "in_process")
        self.assertEqual(todo.get("todo-2").status, "pending")

    def test_claim_is_atomic_and_legacy_view_is_compatible(self) -> None:
        todo = TodoList()
        todo.create(["One item"])
        barrier = threading.Barrier(2)

        def claim() -> str:
            barrier.wait()
            try:
                todo.claim("todo-1")
            except ValueError:
                return "rejected"
            return "claimed"

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: claim(), range(2)))

        self.assertEqual(sorted(results), ["claimed", "rejected"])
        self.assertEqual(todo.get("todo-1").status, "in_process")
        self.assertEqual(todo.items[0].status, "in_progress")

    def test_claim_next_requires_unowned_ready_task_and_sets_owner(self) -> None:
        todo = TodoList()
        todo.create(
            [
                "First",
                {"task": "Second", "dependencies": ["todo-1"]},
            ]
        )

        first = todo.claim_next("worker-a")
        self.assertIsNotNone(first)
        self.assertEqual(first["owner"], "worker-a")
        self.assertFalse(todo.can_start("todo-2"))

        with self.assertRaises(ValueError):
            todo.claim("todo-1", owner="worker-b")

        todo.complete("todo-1", "first finished")
        second = todo.claim_next("worker-b")
        self.assertEqual(second["task_id"], "todo-2")
        self.assertEqual(todo.get("todo-2").owner, "worker-b")

    def test_concurrent_claim_next_assigns_distinct_tasks(self) -> None:
        todo = TodoList()
        todo.create(["First", "Second"])

        with ThreadPoolExecutor(max_workers=2) as executor:
            claimed = list(executor.map(todo.claim_next, ["worker-a", "worker-b"]))

        claimed_ids = [task["task_id"] for task in claimed if task is not None]
        self.assertEqual(sorted(claimed_ids), ["todo-1", "todo-2"])
        self.assertEqual(
            {todo.get(task_id).owner for task_id in claimed_ids},
            {"worker-a", "worker-b"},
        )

    def test_toolset_registers_explicit_actions(self) -> None:
        names = {spec.name for spec in TodoToolset().specs}
        self.assertTrue({"todo_claim", "todo_complete", "todo_block"}.issubset(names))
        background_names = {
            spec.name
            for spec in TodoToolset(background_handler=lambda **_: None).specs
        }
        self.assertIn("todo_run_background", background_names)

    def test_long_install_requests_are_complex_tasks(self) -> None:
        self.assertTrue(is_complex_task("请运行 npm install"))
        self.assertTrue(is_complex_task("安装依赖"))


if __name__ == "__main__":
    unittest.main()
