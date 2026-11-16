"""端到端测试：乱序重投、幂等入库、冲突待核、观测撤下、重启与并发。"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import create_app  # noqa: E402

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")


def make_client(tmp_path, name="t.db"):
    app = create_app(str(tmp_path / name))
    return TestClient(app)


def upload(client, path):
    with open(path, "rb") as fh:
        return client.post("/api/upload", files={"file": (os.path.basename(path), fh)})


def msg_map(client):
    return {(m["boot_id"], m["msg_id"]): m
            for m in client.get("/api/messages").json()["messages"]}


# ---------- 正常流：乱序 + 重投 ----------

def test_out_of_order_and_retransmit(tmp_path):
    with make_client(tmp_path) as c:
        r = upload(c, os.path.join(SAMPLES, "sample_upload.ndjson"))
        assert r.status_code == 200
        counts = r.json()["counts"]
        assert counts["published"] == 2          # 1001、1002 收齐发布
        assert counts["duplicate"] == 3          # 三条完全相同的重投被幂等忽略
        assert counts["conflict"] == 0

        msgs = msg_map(c)
        assert msgs[(7, 1001)]["status"] == "published"
        assert msgs[(7, 1002)]["status"] == "published"
        m3 = msgs[(7, 1003)]
        assert m3["status"] == "collecting"
        assert m3["missing_segments"] == [2]     # 网页缺段列

        obs = c.get("/api/observations", params={"limit": 5000}).json()["observations"]
        assert len(obs) == 120                   # 两个报文各 60 分钟
        assert all(o["observed_at"].endswith(":00Z") for o in obs)

        csv = c.get("/api/observations.csv")
        assert csv.status_code == 200
        assert "attachment" in csv.headers["content-disposition"]
        assert len(csv.text.strip().splitlines()) == 1 + 120


def test_reupload_is_idempotent(tmp_path):
    path = os.path.join(SAMPLES, "sample_upload.ndjson")
    with make_client(tmp_path) as c:
        upload(c, path)
        before = c.get("/api/observations", params={"limit": 5000}).json()["observations"]
        r = upload(c, path)                      # 原样再传一遍
        counts = r.json()["counts"]
        assert counts["stored"] == 0 and counts["published"] == 0
        assert counts["duplicate"] == 12         # 12 行全部命中已入库的段
        after = c.get("/api/observations", params={"limit": 5000}).json()["observations"]
        assert before == after                   # 不重复入库


# ---------- 冲突 -> 待核 ----------

def test_conflicts_flag_pending_review(tmp_path):
    with make_client(tmp_path) as c:
        r = upload(c, os.path.join(SAMPLES, "sample_conflicts.ndjson"))
        assert r.status_code == 200
        msgs = msg_map(c)
        for msg_id, needle in [
            (2001, "内容冲突"), (2002, "总段数冲突"),
            (2003, "摘要校验不符"), (2005, "摘要冲突"),
        ]:
            m = msgs[(7, msg_id)]
            assert m["status"] == "pending_review", msg_id
            assert needle in m["reason"], (msg_id, m["reason"])


def test_published_observations_withdrawn_on_conflict(tmp_path):
    with make_client(tmp_path) as c:
        upload(c, os.path.join(SAMPLES, "sample_conflicts.ndjson"))
        m = msg_map(c)[(7, 2004)]                # 先发布后收到篡改段
        assert m["status"] == "pending_review"
        assert "内容冲突" in m["reason"]
        assert m["observations"] == 0            # 该报文观测已撤下
        obs = c.get("/api/observations", params={"limit": 5000}).json()["observations"]
        assert all(o["msg_id"] != 2004 for o in obs)


def test_pending_message_ignores_late_segments(tmp_path):
    with make_client(tmp_path) as c:
        upload(c, os.path.join(SAMPLES, "sample_conflicts.ndjson"))
        # 待核报文 2001 再来一段：忽略，不入库
        rec = {"station_id": 101, "boot_id": 7, "msg_id": 2001, "seg_no": 1,
               "seg_total": 2, "payload_b64": base64.b64encode(b"x").decode(),
               "sha256": "0" * 64}
        r = c.post("/api/upload", content=json.dumps(rec) + "\n")
        assert r.json()["counts"]["ignored"] == 1
        # 复位后可重新采集
        assert c.delete("/api/messages/101/7/2001").status_code == 200
        assert (7, 2001) not in msg_map(c)


# ---------- 坏行不拖垮整批 ----------

def test_bad_lines_do_not_abort_batch(tmp_path):
    good = {"station_id": 9, "boot_id": 1, "msg_id": 1, "seg_no": 1, "seg_total": 1,
            "payload_b64": base64.b64encode(b"2026-11-15T00:00:00Z,1.23\n").decode(),
            "sha256": hashlib.sha256(b"2026-11-15T00:00:00Z,1.23\n").hexdigest()}
    text = "not json\n" + json.dumps({"station_id": 9}) + "\n" + json.dumps(good) + "\n"
    with make_client(tmp_path) as c:
        r = c.post("/api/upload", content=text)
        counts = r.json()["counts"]
        assert counts["error"] == 2
        assert counts["published"] == 1
        obs = c.get("/api/observations").json()["observations"]
        assert len(obs) == 1 and obs[0]["level_m"] == 1.23


# ---------- 重启 ----------

def test_restart_keeps_state_and_stays_idempotent(tmp_path):
    path = os.path.join(SAMPLES, "sample_upload.ndjson")
    with make_client(tmp_path) as c1:
        upload(c1, path)
        before = c1.get("/api/observations", params={"limit": 5000}).json()["observations"]
    # 模拟服务重启：同一数据库文件起新应用
    with make_client(tmp_path) as c2:
        after = c2.get("/api/observations", params={"limit": 5000}).json()["observations"]
        assert before == after                   # 重启后观测仍在，且只含完整报文
        counts = upload(c2, path).json()["counts"]
        assert counts["stored"] == 0             # 重投不重复入库
        msgs = msg_map(c2)
        assert msgs[(7, 1003)]["missing_segments"] == [2]


# ---------- 并发上传 ----------

def test_concurrent_uploads_no_duplicates(tmp_path):
    path = os.path.join(SAMPLES, "sample_upload.ndjson")
    with make_client(tmp_path) as c:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: upload(c, path), range(8)))
        assert all(r.status_code == 200 for r in results)
        obs = c.get("/api/observations", params={"limit": 5000}).json()["observations"]
        assert len(obs) == 120                   # 恰好入库一次
        keys = [(o["observed_at"]) for o in obs]
        assert len(keys) == len(set(keys))
        msgs = msg_map(c)
        assert msgs[(7, 1001)]["received"] == 4  # 段不重复
        assert msgs[(7, 1002)]["received"] == 3


def test_concurrent_same_single_segment_message(tmp_path):
    # 多个请求同时发同一个单段报文：只发布一次、观测不重复
    payload = b"2026-11-15T10:00:00Z,2.5\n2026-11-15T10:01:00Z,2.6\n"
    rec = json.dumps({"station_id": 5, "boot_id": 2, "msg_id": 9, "seg_no": 1,
                      "seg_total": 1, "payload_b64": base64.b64encode(payload).decode(),
                      "sha256": hashlib.sha256(payload).hexdigest()}) + "\n"
    with make_client(tmp_path) as c:
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda _: c.post("/api/upload", content=rec), range(6)))
        obs = c.get("/api/observations", params={"limit": 5000}).json()["observations"]
        assert len(obs) == 2
        assert msg_map(c)[(2, 9)]["status"] == "published"


# ---------- 单行数据库异常：回滚该行、不拖垮同批、重投可恢复 ----------

class _FlakyConnection:
    """按调用次数注入故障的 sqlite 连接代理。

    fail_on: 'obs_insert'（写观测 executemany）或 'obs_delete'（撤观测 DELETE）。
    第 fail_on_call 次命中时抛 OperationalError，之后恢复正常。
    """

    def __init__(self, real, fail_on, fail_on_call=1):
        self._real = real
        self._fail_on = fail_on
        self._fail_on_call = fail_on_call
        self._calls = 0

    def _maybe_fail(self, kind):
        if kind != self._fail_on:
            return
        self._calls += 1
        if self._calls == self._fail_on_call:
            raise sqlite3.OperationalError(f"injected disk I/O error ({kind})")

    def execute(self, sql, *args):
        if sql.lstrip().upper().startswith("DELETE FROM OBSERVATIONS"):
            self._maybe_fail("obs_delete")
        return self._real.execute(sql, *args)

    def executemany(self, sql, *args):
        self._maybe_fail("obs_insert")
        return self._real.executemany(sql, *args)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _single_segment_record(station, boot, msg, payload):
    return json.dumps({
        "station_id": station, "boot_id": boot, "msg_id": msg,
        "seg_no": 1, "seg_total": 1,
        "payload_b64": base64.b64encode(payload).decode(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }) + "\n"


def test_failed_line_rolls_back_while_other_lines_commit(tmp_path):
    # 同批两个单段报文：第一个写观测时数据库异常 -> 该行全部回滚，第二个照常发布
    with make_client(tmp_path) as c:
        real = c.app.state.db
        c.app.state.db = _FlakyConnection(real, "obs_insert", fail_on_call=1)
        payload1 = b"2026-11-15T00:00:00Z,1.0\n"
        payload2 = b"2026-11-15T01:00:00Z,2.0\n"
        batch = _single_segment_record(11, 3, 100, payload1) + \
            _single_segment_record(11, 3, 200, payload2)
        r = c.post("/api/upload", content=batch)
        c.app.state.db = real
        counts = r.json()["counts"]
        assert counts["error"] == 1 and counts["published"] == 1

        msgs = msg_map(c)
        assert (3, 100) not in msgs                     # 失败行的报文行已回滚
        assert msgs[(3, 200)]["status"] == "published"  # 同批其他行仍可提交
        assert real.execute("SELECT COUNT(*) c FROM segments").fetchone()["c"] == 1
        obs = c.get("/api/observations", params={"limit": 5000}).json()["observations"]
        assert [o["msg_id"] for o in obs] == [200]

        # 故障排除后原样重投：不再报重复，100 正常发布；200 幂等命中
        r2 = c.post("/api/upload", content=batch)
        counts2 = r2.json()["counts"]
        assert counts2["published"] == 1 and counts2["duplicate"] == 1
        assert counts2["error"] == 0
        msgs = msg_map(c)
        assert msgs[(3, 100)]["status"] == "published"
        assert msgs[(3, 100)]["observations"] == 1
        obs = c.get("/api/observations", params={"limit": 5000}).json()["observations"]
        assert sorted(o["msg_id"] for o in obs) == [100, 200]
        # 浏览器预览与 CSV 反映同一份已提交结果
        csv_lines = c.get("/api/observations.csv").text.strip().splitlines()
        assert len(csv_lines) == 1 + 2
        assert {line.split(",")[2] for line in csv_lines[1:]} == {"1.0", "2.0"}


def test_failed_observation_insert_rolls_back_segment_of_multiseg_message(tmp_path):
    # 多段报文最后一段发布时写观测失败：段与报文状态回到处理前，重投可发布
    with make_client(tmp_path) as c:
        real = c.app.state.db
        full = b"# note\n2026-11-15T00:00:00Z,3.1\n2026-11-15T00:01:00Z,3.2\n"
        digest = hashlib.sha256(full).hexdigest()
        seg1 = {"station_id": 12, "boot_id": 4, "msg_id": 90, "seg_no": 1,
                "seg_total": 2, "payload_b64": base64.b64encode(b"# note\n").decode(),
                "sha256": digest}
        seg2 = {"station_id": 12, "boot_id": 4, "msg_id": 90, "seg_no": 2,
                "seg_total": 2, "payload_b64": base64.b64encode(full.split(b"# note\n")[1]).decode(),
                "sha256": digest}
        assert c.post("/api/upload", content=json.dumps(seg1) + "\n").status_code == 200

        c.app.state.db = _FlakyConnection(real, "obs_insert", fail_on_call=1)
        r = c.post("/api/upload", content=json.dumps(seg2) + "\n")
        c.app.state.db = real
        assert r.json()["counts"]["error"] == 1
        m = msg_map(c)[(4, 90)]
        assert m["status"] == "collecting"             # 状态回到处理前
        assert m["received"] == 1 and m["missing_segments"] == [2]
        assert real.execute("SELECT COUNT(*) c FROM segments").fetchone()["c"] == 1

        # 原样重投第二段：发布成功
        r2 = c.post("/api/upload", content=json.dumps(seg2) + "\n")
        assert r2.json()["counts"]["published"] == 1
        assert msg_map(c)[(4, 90)]["status"] == "published"
        obs = c.get("/api/observations", params={"limit": 5000}).json()["observations"]
        assert len(obs) == 2


def test_failed_withdrawal_rolls_back_pending_and_retransmit_withdraws(tmp_path):
    # 已发布报文收到篡改段时撤观测失败 -> 仍保持 published 且观测保留；
    # 故障排除后重投篡改段 -> 转待核并成功撤下，重投不再被 ignored 卡住
    with make_client(tmp_path) as c:
        real = c.app.state.db
        payload = b"2026-11-15T00:00:00Z,4.0\n2026-11-15T00:01:00Z,4.1\n"
        good = _single_segment_record(13, 5, 300, payload)
        assert c.post("/api/upload", content=good).json()["counts"]["published"] == 1

        forged = dict(json.loads(good))
        forged["payload_b64"] = base64.b64encode(b"# forged\n").decode()
        c.app.state.db = _FlakyConnection(real, "obs_delete", fail_on_call=1)
        r = c.post("/api/upload", content=json.dumps(forged) + "\n")
        c.app.state.db = real
        assert r.json()["counts"]["error"] == 1

        m = msg_map(c)[(5, 300)]
        assert m["status"] == "published"               # 撤回失败不残留待核状态
        assert m["observations"] == 2                   # 观测仍在
        obs = c.get("/api/observations", params={"limit": 5000}).json()["observations"]
        assert len(obs) == 2

        # 重投篡改段：这次撤下成功，转待核
        r2 = c.post("/api/upload", content=json.dumps(forged) + "\n")
        assert r2.json()["counts"]["conflict"] == 1
        m = msg_map(c)[(5, 300)]
        assert m["status"] == "pending_review"
        assert "内容冲突" in m["reason"]
        assert m["observations"] == 0
        assert c.get("/api/observations", params={"limit": 5000}).json()["observations"] == []
        # CSV 同样反映撤下后的结果
        assert c.get("/api/observations.csv").text.strip() == "station_id,observed_at,level_m"

        # 复位后可重新采集（原有待核复位规则不受影响）
        assert c.delete("/api/messages/13/5/300").status_code == 200
        assert (5, 300) not in msg_map(c)
