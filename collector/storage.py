"""磁盘布局与分片/发布文件管理。

DATA_DIR/
├── collector.db
├── locks/<task_id>.lock        # 跨实例 fcntl 互斥
├── parts/<task_id>/r<rev>.part # 未完成分片（按修订号隔离）
├── tmp/...                     # 发布临时文件（原子 replace 前）
└── published/...               # 已发布、可下载的一致原件

清理纪律：发布临时文件只在“创建”（publish_atomic，持任务锁）与
“清理”（cleanup_tmp / cleanup_tmp_for_task，同样持任务锁）两侧都受锁保护，
因此恢复检查与继续采集交错时，绝不会按过时判断删掉另一实例刚建的新文件。
分片（parts/）不属于清理范围，一律保留供断点续传。
"""
from __future__ import annotations

import os

from .locks import TaskLockedError, task_lock


class Storage:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.locks_dir = os.path.join(data_dir, "locks")
        self.parts_dir = os.path.join(data_dir, "parts")
        self.tmp_dir = os.path.join(data_dir, "tmp")
        self.published_dir = os.path.join(data_dir, "published")
        for d in (self.locks_dir, self.parts_dir, self.tmp_dir, self.published_dir):
            os.makedirs(d, exist_ok=True)

    def db_path(self) -> str:
        return os.path.join(self.data_dir, "collector.db")

    def lock_path(self, task_id: str) -> str:
        return os.path.join(self.locks_dir, f"{task_id}.lock")

    def part_path(self, task_id: str, revision: int) -> str:
        d = os.path.join(self.parts_dir, task_id)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"r{revision}.part")

    def final_path(self, station: str, date: str, revision: int) -> str:
        safe_station = _safe(station)
        safe_date = _safe(date)
        return os.path.join(
            self.published_dir, f"{safe_station}_{safe_date}_r{revision}.csv"
        )

    def tmp_path(self, task_id: str, revision: int) -> str:
        return os.path.join(self.tmp_dir, f"{task_id}-r{revision}-{os.getpid()}.tmp")

    def cleanup_tmp(self) -> None:
        """启动时清理发布临时文件残件（非阻塞，不等待活跃采集结束）。

        逐任务试探任务锁：
        - 锁被持有 → 另一实例正在采集/发布该任务，其临时文件绝不动；
        - 锁可立即获得 → 属主已退出，残件在锁内删除（创建与删除同锁，
          不会误删活跃实例刚建立的文件）。
        """
        try:
            names = os.listdir(self.tmp_dir)
        except FileNotFoundError:
            return
        task_ids = {
            task_id
            for name in names
            if (task_id := tmp_task_id(name)) is not None
        }
        for task_id in task_ids:
            try:
                with task_lock(self.lock_path(task_id)):
                    self._unlink_tmp_locked(task_id)
            except TaskLockedError:
                continue

    def cleanup_tmp_for_task(self, task_id: str) -> None:
        """清理指定任务的发布临时残件。调用方必须已持有该任务锁。"""
        self._unlink_tmp_locked(task_id)

    def _unlink_tmp_locked(self, task_id: str) -> None:
        # 在锁内重新列目录：锁保证此刻没有属主在创建该任务的新临时文件
        try:
            names = os.listdir(self.tmp_dir)
        except FileNotFoundError:
            return
        for name in names:
            if tmp_task_id(name) != task_id:
                continue
            try:
                os.unlink(os.path.join(self.tmp_dir, name))
            except FileNotFoundError:
                pass

    def publish_atomic(self, part: str, final: str) -> None:
        """硬链接到临时名再原子 replace；同文件系统，不复制大文件。"""
        tmp = self.tmp_path(_basename_task(part), _basename_rev(part))
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        os.link(part, tmp)
        os.replace(tmp, final)
        _fsync_dir(self.published_dir)


def tmp_task_id(name: str) -> str | None:
    """从发布临时文件名 '<task_id>-r<rev>-<pid>.tmp' 解析任务 ID；不匹配返回 None。"""
    if not name.endswith(".tmp"):
        return None
    head, _, _ = name.partition("-")
    return head if head.isdigit() else None


def _safe(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value)


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _basename_task(part_path: str) -> str:
    # parts/<task_id>/r<rev>.part
    return os.path.basename(os.path.dirname(part_path))


def _basename_rev(part_path: str) -> int:
    name = os.path.basename(part_path)
    return int(name[1:].split(".", 1)[0])
