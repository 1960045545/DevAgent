# 子 Agent 委派

可以把互不依赖的 Task 节点交给 `todo_delegate`。只传递该节点的明确说明和完成它所需的最小工具能力。子 Agent 使用全新的 messages，不继承主 Agent 的对话。委派前节点必须由主 Agent 通过 `todo_claim` 进入 `in_process`；只有收到 completed 摘要后才调用 `todo_complete` 写回 Task 的 `summary`。blocked 或 failed 必须保留原状态和原因，依赖它们的 pending 节点会被标记为 blocked。

## 文件式协作

每次复杂请求都有独立的通信目录。主 Agent 与子 Agent 通过通信工具交换结构化 JSON 消息、共享上下文文件和任务产物，不通过复制主 Agent 的历史消息来共享上下文。

1. 委派时系统会保存任务的 `input.json`；子 Agent 完成后系统会保存 `result.json`，完整记录可溯源。
2. 子 Agent 开始依赖任务前，必须调用 `subagent_read_task_result` 读取已完成前置任务的结果；需要更长内容时调用 `subagent_read_artifact` 或 `subagent_read_shared_context`。
3. 任务委派、上下文请求、产物交付、计划审批和关闭请求必须使用 `subagent_request_protocol`；响应必须使用 `subagent_respond_protocol`，并引用收到的 `request_id`。协议 payload 必须是 JSON 对象，不能用自由文本替代必填字段。
4. 主 Agent 和子 Agent 读取消息时统一调用 `subagent_consume_inbox`。每条消息只会被消费一次；协议响应只有在收件方消费后才会更新协议卡片状态。`subagent_get_protocol` 可用于查询状态。
5. 较大的内容写入 `subagent_write_artifact` 或 `subagent_write_shared_context`，在协议 payload 的 `artifacts` 或消息内容中引用请求范围内的路径。
6. 所有通信工具都限制在当前用户工作区和当前请求目录内，禁止绝对路径、路径穿越和敏感文件访问。不要使用普通 shell 代替通信工具写入通信目录。

## 空闲 Agent 自主取任务

子 Agent 完成当前任务后会进入空闲轮询状态：每 5 秒先消费自己的 inbox，再扫描共享 Todo 看板。只有同时满足 `status=pending`、`owner` 为空、`can_start=true` 的任务才会被原子认领，并将 owner 写为当前 Agent；空闲超过配置的超时时间后自动退出。收到 `shutdown_request` 时必须立即确认并停止。

## Git worktree 隔离

每个子 Agent 任务都必须在独立的 Git worktree 中执行。worktree 位于当前用户工作区的 `.worktrees/<name>`，并使用 `wt/<name>` 分支；任务会在开始执行前绑定 worktree。子 Agent 的文件、Shell 和 Python 工具都只允许访问绑定 worktree，不得访问父工作区或其它 worktree。任务结束后 worktree 不会自动删除，必须保留给用户选择：调用 `worktree_keep` 保留目录和分支以便检查，或调用 `worktree_remove` 删除；检测到未提交修改时，删除必须显式传入 `discard_changes=true`。不得替用户猜测保留或删除决定，也不得自动调用这两个清理工具；应把路径、分支和脏状态报告给用户，等待用户明确选择。
