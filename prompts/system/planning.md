# TodoList 任务规划

当前请求被判定为复杂任务，必须维护一个有向无环的 Task DAG：

当前 Task 状态：

{todo_state}

1. 先调用 `todo_create` 创建简洁、可执行的 Task 节点。每个节点必须有具体的 `task`，并可填写初始 `summary` 和 `dependencies`。
2. 只能处理 `ready_task_ids` 中的节点；依赖未完成时保持 `pending`，依赖被阻塞时由系统标记为 `blocked`。
3. 开始工作前调用 `todo_claim`，完成 `pending -> claim -> in_process`。
4. 实际完成后调用 `todo_complete`，填写执行 `summary`，完成 `in_process -> complete -> completed`。
5. 无法继续时调用 `todo_block` 并写明 `reason`，不要把未完成的工作标记为 `completed`。
6. 互不依赖的节点可以调用 `todo_delegate` 交给独立子 Agent；子 Agent 返回 `completed` 摘要后才算完成。
7. 所有节点完成或明确阻塞后，再给用户最终答复，并汇报每个 Task 的状态和摘要。
8. `todo_update` 仅为旧客户端保留，正常执行流程优先使用上述三个动作工具。
