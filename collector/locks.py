"""跨实例任务互斥锁（fcntl.flock，非阻塞）。

锁是“任务是否有活跃属主”的唯一事实来源：

- 采集/发布全程持锁（见 transfer.run_resume）；
- 启动恢复与临时文件清理用 LOCK_NB 试探：
  - 拿不到锁 → 另一实例正在采集/发布，绝不改动其状态、绝不清理其文件；
  - 拿得到锁 → 属主已退出（进程消亡时内核自动释放 flock），可安全回收。
"""
from __future__ import annotations

import contextlib
import fcntl
import os
from dataclasses import dataclass
from typing import Iterator


class TaskLockedError(Exception):
    """另一实例/线程正在处理该任务。"""


@dataclass
class LockHandle:
    fd: int
    path: str


@contextlib.contextmanager
def task_lock(lock_path: str) -> Iterator[LockHandle]:
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise TaskLockedError(lock_path) from exc
        yield LockHandle(fd=fd, path=lock_path)
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
