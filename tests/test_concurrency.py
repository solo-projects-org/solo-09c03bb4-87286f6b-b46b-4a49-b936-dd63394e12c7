"""并发与崩溃安全：双实例锁、锁感知启动恢复、发布原子性、下载防半件。"""
from __future__ import annotations

import os
import threading

import pytest

from collector import db as schema
from collector.config import Settings
from collector.transfer import TaskLockedError, run_resume, task_lock


def test_task_lock_is_mutually_exclusive(tmp_path):
    lock = str(tmp_path / "x.lock")
    with task_lock(lock):
        with pytest.raises(TaskLockedError):
            with task_lock(lock):
                pass


def test_two_instances_concurrent_resume_one_waits_rejected(env):
    results: list[str] = []
    barrier = threading.Barrier(2)

    def worker(instance: str) -> None:
        settings = Settings(**{**dataclasses_asdict(env.settings), "instance_id": instance})
        barrier.wait()
        try:
            run_resume(env.db, env.storage, settings, env.task_id)
            results.append("ok")
        except TaskLockedError:
            results.append("locked")
        except Exception as exc:  # pragma: no cover
            results.append(f"error:{exc}")

    t1 = threading.Thread(target=worker, args=("inst-1",))
    t2 = threading.Thread(target=worker, args=("inst-2",))
    t1.start(); t2.start(); t1.join(); t2.join()

    assert sorted(results) == ["locked", "ok"]
    detail = env.db.task_detail(env.task_id)
    assert detail["state"] == schema.TASK_PUBLISHED
    # 只有一个实例登记为完成者；分片不存在交错写入
    with open(detail["latest_published"]["file_path"], "rb") as fh:
        import hashlib
        assert hashlib.sha256(fh.read()).hexdigest() == detail["latest_published"]["sha256"]


def dataclasses_asdict(settings: Settings) -> dict:
    import dataclasses
    return dataclasses.asdict(settings)


def test_restart_recovers_downloading_state_to_interrupted(env):
    # 制造一次中断，状态为 DOWNLOADING 风格（模拟进程在写库后、收尾前被杀）
    env.source.drop_after = 500
    with pytest.raises(Exception):
        env.resume()

    # 手工把状态拨回 DOWNLOADING，模拟崩溃瞬间
    import sqlite3
    conn = sqlite3.connect(env.storage.db_path())
    conn.execute(
        f"UPDATE tasks SET state='{schema.TASK_DOWNLOADING}' WHERE id=?",
        (env.task_id,),
    )
    conn.execute(
        f"UPDATE task_versions SET state='{schema.REV_DOWNLOADING}' WHERE task_id=?",
        (env.task_id,),
    )
    conn.commit(); conn.close()

    # 新进程重新打开库：启动恢复把 DOWNLOADING 归位为 INTERRUPTED，绝不误报完成
    fresh_db = schema.Database(env.storage.db_path())
    detail = fresh_db.task_detail(env.task_id)
    assert detail["state"] == schema.TASK_INTERRUPTED
    assert detail["active_version"]["state"] == schema.REV_INTERRUPTED
    # 分片仍在，恢复后可继续
    detail = run_resume(fresh_db, env.storage, env.settings, env.task_id)
    assert detail["state"] == schema.TASK_PUBLISHED


def test_tmp_leftover_cleaned_on_startup(env):
    leftover = env.storage.tmp_path(str(env.task_id), 1)
    with open(leftover, "wb") as fh:
        fh.write(b"x")
    # 应用启动流程中的清理动作（无活跃属主，残件清除）
    env.storage.cleanup_tmp()
    assert not os.path.exists(leftover)


def test_published_file_remains_available_after_re_interrupt(env):
    detail = env.resume()
    published = detail["latest_published"]["file_path"]
    assert os.path.isfile(published)

    # 换版后新修订下载中断：任务整体仍可通过 latest_published 取到旧版
    env.source.bump()
    env.source.drop_after = 100
    with pytest.raises(Exception):
        env.resume()
    detail = env.db.task_detail(env.task_id)
    assert detail["latest_published"]["revision"] == 1
    assert os.path.isfile(published)


def test_publish_is_idempotent_under_repeated_call(env):
    env.resume()
    detail1 = env.db.task_detail(env.task_id)
    # 再次 resume：版本已 PUBLISHED，直接返回，不产生重复发布
    detail2 = env.resume()
    assert detail1["latest_published"]["file_path"] == detail2["latest_published"]["file_path"]
    import sqlite3
    conn = sqlite3.connect(env.storage.db_path())
    n = conn.execute(
        "SELECT COUNT(*) FROM publishes WHERE task_id=?", (env.task_id,)
    ).fetchone()[0]
    conn.close()
    assert n == 1


# ---------- 轮换重启：锁感知启动恢复 ----------

def _force_downloading_state(env) -> None:
    """把任务/版本拨到 DOWNLOADING，模拟实例正在采集/发布的瞬间。"""
    import sqlite3
    conn = sqlite3.connect(env.storage.db_path())
    conn.execute(
        f"UPDATE tasks SET state='{schema.TASK_DOWNLOADING}' WHERE id=?",
        (env.task_id,),
    )
    conn.execute(
        f"UPDATE task_versions SET state='{schema.REV_DOWNLOADING}' WHERE task_id=?",
        (env.task_id,),
    )
    conn.commit()
    conn.close()


def test_startup_recovery_skips_locked_task_and_recovers_after_exit(env):
    # 先制造一次中断，留下分片
    env.source.drop_after = 500
    with pytest.raises(Exception):
        env.resume()
    part = env.part_path(1)
    saved = os.path.getsize(part)
    assert saved > 0
    _force_downloading_state(env)

    lock_path = env.storage.lock_path(str(env.task_id))
    with task_lock(lock_path):
        # 实例 B 启动：另一实例仍持锁采集，恢复绝不动其状态
        fresh_db = schema.Database(env.storage.db_path(), locks_dir=env.storage.locks_dir)
        detail = fresh_db.task_detail(env.task_id)
        assert detail["state"] == schema.TASK_DOWNLOADING
        assert detail["active_version"]["state"] == schema.REV_DOWNLOADING

    # 属主退出（锁随进程消亡释放）后再启动：现场归位 INTERRUPTED，分片保留
    fresh_db2 = schema.Database(env.storage.db_path(), locks_dir=env.storage.locks_dir)
    detail = fresh_db2.task_detail(env.task_id)
    assert detail["state"] == schema.TASK_INTERRUPTED
    assert detail["active_version"]["state"] == schema.REV_INTERRUPTED
    assert os.path.getsize(part) == saved
    # 恢复后可续传完成
    detail = run_resume(fresh_db2, env.storage, env.settings, env.task_id)
    assert detail["state"] == schema.TASK_PUBLISHED


def test_cleanup_tmp_skips_locked_task_and_cleans_orphans(env):
    # 活跃任务的发布临时文件（另一实例正在发布）
    active_tmp = env.storage.tmp_path(str(env.task_id), 1)
    with open(active_tmp, "wb") as fh:
        fh.write(b"publishing")
    # 已退出实例的残件（无属主任务）
    orphan_tmp = env.storage.tmp_path("999999", 3)
    with open(orphan_tmp, "wb") as fh:
        fh.write(b"stale")

    lock_path = env.storage.lock_path(str(env.task_id))
    with task_lock(lock_path):
        env.storage.cleanup_tmp()
        assert os.path.exists(active_tmp)       # 活跃实例的文件未被触碰
        assert not os.path.exists(orphan_tmp)   # 无属主残件被清理

    # 锁释放后再清理：残件可回收
    env.storage.cleanup_tmp()
    assert not os.path.exists(active_tmp)


def test_publish_in_progress_survives_other_instance_startup(env):
    """Bug 现场：A 已收齐、正在发布（tmp 已 link），B 启动不得删其临时文件、
    不得改其任务状态，A 随后的原子替换必须成功。"""
    env.source.drop_after = 400
    with pytest.raises(Exception):
        env.resume()
    # 补全分片为完整一致原件（模拟 A 已收齐）
    blob = env.source.blobs[1]
    part = env.part_path(1)
    with open(part, "wb") as fh:
        fh.write(blob)
    _force_downloading_state(env)

    lock_path = env.storage.lock_path(str(env.task_id))
    with task_lock(lock_path):
        # A 发布中：硬链接出临时文件，尚未 os.replace
        tmp = env.storage.tmp_path(str(env.task_id), 1)
        os.link(part, tmp)

        # 实例 B 启动：恢复 + 清理都不得触碰活跃任务
        fresh_db = schema.Database(env.storage.db_path(), locks_dir=env.storage.locks_dir)
        env.storage.cleanup_tmp()
        assert os.path.exists(tmp), "活跃实例的发布临时文件被误删"
        detail = fresh_db.task_detail(env.task_id)
        assert detail["state"] == schema.TASK_DOWNLOADING
        assert detail["active_version"]["state"] == schema.REV_DOWNLOADING

        # A 继续完成发布：原子替换不再报错
        final = env.storage.final_path("ST01", "2026-11-16", 1)
        os.replace(tmp, final)
        assert os.path.isfile(final)


def test_dead_instance_publish_leftover_cleaned_part_kept_no_fake_completion(env):
    """已退出实例的发布残件：tmp 自动清理、分片保留、状态归位 INTERRUPTED，
    恢复绝不把“看似收齐”的版本登记为完成（无 publishes 记录）。"""
    blob = env.source.blobs[1]
    env.source.drop_after = 400
    with pytest.raises(Exception):
        env.resume()  # 建立版本行与分片
    part = env.part_path(1)
    with open(part, "wb") as fh:
        fh.write(blob)  # 完整分片：属主在发布前一刻退出
    _force_downloading_state(env)
    stale_tmp = env.storage.tmp_path(str(env.task_id), 1)
    os.link(part, stale_tmp)

    # 实例 B 启动（属主已退出，锁可立即获得）
    fresh_db = schema.Database(env.storage.db_path(), locks_dir=env.storage.locks_dir)
    env.storage.cleanup_tmp()

    assert not os.path.exists(stale_tmp), "已退出实例的发布残件未清理"
    assert os.path.getsize(part) == len(blob), "分片必须保留供续传"
    detail = fresh_db.task_detail(env.task_id)
    assert detail["state"] == schema.TASK_INTERRUPTED
    assert detail["active_version"]["state"] == schema.REV_INTERRUPTED
    assert detail["latest_published"] is None, "恢复不得登记缺件的完成记录"

    # 续传走正常流程：重新校验后发布，字节与源站一致
    detail = run_resume(fresh_db, env.storage, env.settings, env.task_id)
    assert detail["state"] == schema.TASK_PUBLISHED
    with open(detail["latest_published"]["file_path"], "rb") as fh:
        assert fh.read() == blob


def test_resume_cleans_stale_publish_leftover_under_lock(env):
    """续传持锁后先清理本任务的发布残件（已退出实例遗留），不影响正常发布。"""
    stale_tmp = env.storage.tmp_path(str(env.task_id), 1)
    with open(stale_tmp, "wb") as fh:
        fh.write(b"leftover-from-dead-instance")
    detail = env.resume()
    assert detail["state"] == schema.TASK_PUBLISHED
    assert not os.path.exists(stale_tmp)
