# 子 Agent 委派

可以把互不依赖的 Task 节点交给 `todo_delegate`。只传递该节点的明确说明和完成它所需的最小工具能力。子 Agent 使用全新的 messages，不继承主 Agent 的对话。委派前节点必须由主 Agent 通过 `todo_claim` 进入 `in_process`；只有收到 completed 摘要后才调用 `todo_complete` 写回 Task 的 `summary`。blocked 或 failed 必须保留原状态和原因，依赖它们的 pending 节点会被标记为 blocked。
