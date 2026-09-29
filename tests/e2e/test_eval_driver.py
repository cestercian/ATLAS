"""The held-out evaluation driver (scripts/eval/), on a synthetic toy suite.

Nothing here comes from the held-out suite: the tasks are made up in each
test, the model and the sandbox are fakes, and docker is never called.
"""
import json
import socket
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "eval"))

import atlas_arm  # noqa: E402
import baseline_arm as B  # noqa: E402
import driver  # noqa: E402
import grading as G  # noqa: E402
import report as R  # noqa: E402
import suite as S  # noqa: E402
from result import ArmResult  # noqa: E402


# --- a toy suite -----------------------------------------------------------------

def make_suite(root: Path, runtime="python3.13", extra=None) -> Path:
    task = root / "tasks" / "t01"
    files = {
        "task.json": json.dumps({"id": "t01", "mode": "work", "runtime": runtime,
                                 "network": False, "grader_timeout_s": 60}),
        "prompt.md": "Write answer.txt containing 42.\n",
        "grade": "#!/bin/sh\ngrep -q 42 answer.txt\n",
        "seed/README": "toy\n",
        "controls/pass/answer.txt": "42\n",
        "controls/fail/answer.txt": "41\n",
    }
    files.update(extra or {})
    for rel, text in files.items():
        (task / rel).parent.mkdir(parents=True, exist_ok=True)
        (task / rel).write_text(text)
    manifest = {"suite": "toy", "tasks": [{"id": "t01", "files": {
        rel: S.sha256_file(task / rel) for rel in files}}]}
    (root / "suite.json").write_text(json.dumps(manifest))
    return root


def test_a_frozen_suite_loads(tmp_path):
    tasks = S.load_suite(make_suite(tmp_path))
    assert [t.id for t in tasks] == ["t01"]
    assert tasks[0].prompt.startswith("Write answer.txt")


def test_a_relative_suite_path_reaches_the_grader_as_an_absolute_one(tmp_path, monkeypatch):
    make_suite(tmp_path / "s")
    monkeypatch.chdir(tmp_path)
    task = S.load_suite(Path("s"))[0]
    argv = G.grader_argv(task, tmp_path / "w", "img")
    mounts = [a.split(":")[0] for a in argv if a.endswith((":/grade:ro", ":/grader:ro"))]
    assert mounts and all(Path(m).is_absolute() for m in mounts)


def test_the_check_gives_the_same_result_for_a_relative_and_an_absolute_path(tmp_path, monkeypatch):
    suite = make_suite(tmp_path / "s")
    monkeypatch.chdir(tmp_path)
    seen = []

    def fake_docker(argv, **kw):
        seen.append(next(a for a in argv if a.endswith(":/grade:ro")))
        return FakeDocker()(argv, **kw)

    monkeypatch.setattr(driver, "check_controls",
                        lambda task, image: G.check_controls(task, image, run=fake_docker))
    assert driver.main(["check", "s", "--image", "img"]) == 0
    assert driver.main(["check", str(suite), "--image", "img"]) == 0
    assert len(seen) == 4 and len(set(seen)) == 1
    assert Path(seen[0].split(":")[0]).is_absolute()


@pytest.mark.parametrize("tamper, want", [
    (lambda t: (t / "prompt.md").write_text("changed\n"), "changed since the freeze: prompt.md"),
    (lambda t: (t / "seed" / "extra.py").write_text("x\n"), "added after the freeze: seed/extra.py"),
    (lambda t: (t / "controls" / "fail" / "answer.txt").unlink(), "missing: controls/fail/answer.txt"),
])
def test_a_suite_that_changed_after_the_freeze_is_refused(tmp_path, tamper, want):
    root = make_suite(tmp_path)
    tamper(root / "tasks" / "t01")
    with pytest.raises(S.SuiteError, match=want):
        S.load_suite(root)


def test_a_runtime_the_sandbox_lacks_is_refused(tmp_path):
    with pytest.raises(S.SuiteError, match="runtime 'cobol'"):
        S.load_suite(make_suite(tmp_path, runtime="cobol"))


# --- grading ---------------------------------------------------------------------

@pytest.mark.parametrize("code, outcome", [(0, G.PASS), (1, G.FAIL), (2, G.GRADER_ERROR),
                                           (124, G.GRADER_ERROR)])
def test_only_exit_0_and_1_grade_the_work(code, outcome):
    assert G.outcome_of(code, "reason\n", "", 60).outcome == outcome


def test_the_grader_runs_sealed(tmp_path):
    task = S.load_suite(make_suite(tmp_path))[0]
    argv = G.grader_argv(task, tmp_path / "copy", "sandbox:x")
    assert argv[:4] == ["docker", "run", "--rm", "--network"] and argv[4] == "none"
    assert f"{task.grader}:/grade:ro" in argv
    assert argv[-4:] == ["sandbox:x", "60", "/grade", "/w"]


class FakeDocker:
    """Grades like the toy grader would: pass when the copy's answer.txt says 42.
    It also writes into the copy, which must never reach the original."""

    def __call__(self, argv, **kw):
        copy = Path(next(a.split(":")[0] for a in argv if a.endswith(":/w")))
        ok = (copy / "answer.txt").is_file() and "42" in (copy / "answer.txt").read_text()
        (copy / "grader-was-here").write_text("x")

        class P:
            returncode = 0 if ok else 1
            stdout = "found 42" if ok else "no 42"
            stderr = ""
        return P()


def test_grading_uses_a_copy_and_leaves_the_workspace_alone(tmp_path):
    task = S.load_suite(make_suite(tmp_path / "s"))[0]
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "answer.txt").write_text("42\n")
    assert G.grade(task, ws, "img", run=FakeDocker()).outcome == G.PASS
    assert not (ws / "grader-was-here").exists()


def test_a_dangling_link_is_graded_as_a_link(tmp_path):
    task = S.load_suite(make_suite(tmp_path / "s"))[0]
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "answer.txt").write_text("42\n")
    (ws / "gone").symlink_to(tmp_path / "nowhere")
    seen = {}

    def docker(argv, **kw):
        copy = Path(next(a.split(":")[0] for a in argv if a.endswith(":/w")))
        seen["link"] = (copy / "gone").is_symlink()
        return FakeDocker()(argv, **kw)
    assert G.grade(task, ws, "img", run=docker).outcome == G.PASS
    assert seen["link"]


def test_a_workspace_that_cannot_be_copied_is_a_grader_error(tmp_path, monkeypatch):
    task = S.load_suite(make_suite(tmp_path / "s"))[0]

    def fails(*a, **k):
        raise G.shutil.Error([("src", "dst", "unreadable")])
    monkeypatch.setattr(G.shutil, "copytree", fails)
    got = G.grade(task, tmp_path, "img", run=FakeDocker())
    assert got.outcome == G.GRADER_ERROR and "could not be copied" in got.reason


def test_controls_that_separate_pass_the_check(tmp_path):
    task = S.load_suite(make_suite(tmp_path))[0]
    assert G.check_controls(task, "img", run=FakeDocker()) == []


def test_controls_that_do_not_separate_are_named(tmp_path):
    root = make_suite(tmp_path, extra={"controls/fail/answer.txt": "42\n"})
    task = S.load_suite(root)[0]
    assert G.check_controls(task, "img", run=FakeDocker()) == [
        "t01: controls/fail graded pass (found 42)"]


# --- the ATLAS arm ---------------------------------------------------------------

def test_the_atlas_arm_reports_the_sessions_own_terminal():
    events = [{"type": "turn_start"}, {"type": "llm_call_end", "data": {"total_tokens": 900}},
              {"type": "turn_start"}, {"type": "done", "data": {"status": "stopped",
                                                               "reason": "repeat_detector"}}]
    r = atlas_arm.result_of(events, 12.3)
    assert (r.status, r.reason, r.turns, r.tokens) == ("stopped", "repeat_detector", 2, 900)


def test_a_done_without_status_is_incomplete_and_no_done_is_its_own_outcome():
    assert atlas_arm.result_of([{"type": "done", "data": {}}], 1).status == "incomplete"
    r = atlas_arm.result_of([{"type": "error", "data": {"error": "stream failed: reset"}}], 1)
    assert (r.status, r.reason) == ("no_terminal", "stream failed: reset")


def test_the_atlas_arm_sends_what_the_tui_sends(tmp_path):
    task = S.load_suite(make_suite(tmp_path))[0]
    sent = {}

    class Resp:
        def __enter__(self):
            return iter([b'data: {"type":"done","data":{"status":"completed","reason":"x"}}\n',
                         b"data: [DONE]\n"])

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout):
        sent.update(json.loads(req.data))
        return Resp()
    r = atlas_arm.run_atlas(task, "http://proxy", "sub1", "sess1", 60, urlopen=urlopen)
    assert r.status == "completed"
    assert sent == {"message": task.prompt, "mode": "yolo", "sandbox_subdir": "sub1",
                    "session_id": "sess1", "task_contract": {"task_mode": "work"}}


# --- the baseline arm ------------------------------------------------------------

def test_the_first_tool_object_is_found_among_prose():
    reply = 'Sure. {"note": 1} then {"tool": "write_file", "path": "a", "content": "{x}"} done'
    assert B.first_json_object(reply) == {"tool": "write_file", "path": "a", "content": "{x}"}
    assert B.first_json_object("no json here") is None


def test_paths_stay_inside_the_workspace(tmp_path):
    ws = B.Workspace(tmp_path, "/workspace/run1")
    assert ws.resolve("a/b.py") == (tmp_path / "a" / "b.py").resolve()
    assert ws.resolve("/workspace/run1/c.py") == (tmp_path / "c.py").resolve()
    assert ws.resolve("../escape.py") is None
    assert ws.resolve("/etc/passwd") is None


class FakeServers:
    """The model replies from a script; the sandbox records each command."""

    def __init__(self, replies):
        self.replies, self.bodies, self.commands = list(replies), [], []

    def __call__(self, url, body, timeout):
        if url.endswith("/shell"):
            self.commands.append(body)
            return {"exit_code": 0, "stdout": "ok\n", "stderr": ""}
        self.bodies.append(body)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {"choices": [{"message": {"content": reply}}], "usage": {"total_tokens": 100}}


def _run(tmp_path, replies, budget=600):
    task = S.load_suite(make_suite(tmp_path / "s"))[0]
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    fake = FakeServers(replies)
    r = B.run_baseline(task, B.Workspace(ws_dir, "/workspace/run1"), "http://llama",
                       "http://sandbox", budget, 32768, post=fake)
    return r, fake, ws_dir


def test_the_baseline_writes_runs_and_finishes(tmp_path):
    r, fake, ws = _run(tmp_path, [
        '{"tool": "write_file", "path": "answer.txt", "content": "42\\n"}',
        '{"tool": "run_command", "command": "cat answer.txt"}',
        '{"tool": "done", "summary": "wrote it"}'])
    assert (r.status, r.turns, r.tokens) == ("completed", 3, 300)
    assert (ws / "answer.txt").read_text() == "42\n"
    assert fake.commands == [{"command": "cat answer.txt", "cwd": "/workspace/run1", "timeout": 30}]


def test_the_baseline_samples_as_atlas_does_with_no_grammar(tmp_path):
    _, fake, _ = _run(tmp_path, ['{"tool": "done", "summary": "."}'])
    body = fake.bodies[0]
    assert body["samplers"] == ["top_k"] and body["top_k"] == 1
    assert body["max_tokens"] == 8192
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "grammar" not in body and "response_format" not in body
    assert body["messages"][0]["content"] == B.SYSTEM_PROMPT


def test_a_reply_without_a_tool_object_is_answered_and_costs_a_turn(tmp_path):
    r, fake, _ = _run(tmp_path, ["I will now think about it.", '{"tool": "done", "summary": "."}'])
    assert r.turns == 2
    assert fake.bodies[1]["messages"][-1]["content"] == B.NO_JSON_REPLY


def test_the_budget_ends_the_run(tmp_path):
    r, _, _ = _run(tmp_path, [socket.timeout("slow")])
    assert (r.status, r.turns) == ("timed_out", 0)
    r, _, _ = _run(tmp_path / "b", ['{"tool": "done", "summary": "."}'], budget=0)
    assert r.status == "timed_out"


def test_a_command_timeout_is_capped_as_atlas_caps_it(tmp_path):
    _, fake, _ = _run(tmp_path, ['{"tool": "run_command", "command": "sleep 1", "timeout": 900}',
                                 '{"tool": "done", "summary": "."}'])
    assert fake.commands[0]["timeout"] == 300


def test_the_oldest_results_give_way_when_the_context_is_full():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"},
            {"role": "assistant", "content": "a1"}, {"role": "user", "content": "x" * 90000},
            {"role": "assistant", "content": "a2"}, {"role": "user", "content": "recent"}]
    fitted = B.fit_context(msgs, 8192 + 20000)
    assert fitted[3]["content"] == "[result omitted: 90000 characters]"
    assert fitted[:2] == msgs[:2] and fitted[5]["content"] == "recent"


# --- the report ------------------------------------------------------------------

def test_wilson_matches_a_known_interval():
    lo, hi = R.wilson(71, 84)
    assert (round(lo, 3), round(hi, 3)) == (0.753, 0.907)


def _rec(task, rep, status, grade, reason="x"):
    return {"task": task, "rep": rep, "status": status, "reason": reason, "grade": grade}


def test_the_summary_counts_false_completions_and_keeps_no_task_text():
    recs = [_rec("t1", 1, "completed", G.PASS), _rec("t1", 2, "completed", G.FAIL),
            _rec("t2", 1, "stopped", G.PASS), _rec("t3", 1, "completed", G.GRADER_ERROR)]
    s = R.arm_summary(recs)
    assert (s["graded"], s["passed"], s["grader_errors"]) == (3, 2, 1)
    assert (s["completed"], s["false_completed"], s["passed_not_completed"]) == (2, 1, 1)
    assert s["tasks_whose_repeats_disagree"] == 1
    assert "t1" not in json.dumps(s) and "t2" not in json.dumps(s)


def test_the_arms_are_compared_with_an_interval():
    a = [_rec(f"t{i}", 1, "completed", G.PASS if i < 8 else G.FAIL) for i in range(10)]
    b = [_rec(f"t{i}", 1, "completed", G.PASS if i < 4 else G.FAIL) for i in range(10)]
    d = R.compare(a, b)["pass_rate_difference"]
    assert d["estimate"] == 0.4 and d["ci95"][0] < 0.4 < d["ci95"][1]


# --- the driver ------------------------------------------------------------------

def test_the_reliability_runner_loads():
    """stack_identity reaches the #241 checks through this loader; the other
    driver tests replace stack_identity, so none of them load the module."""
    rel = driver._reliability_runner()
    assert callable(rel.deployed_identity) and callable(rel.stack_identity)


def test_the_development_stack_is_never_measured(tmp_path):
    args = driver.parse_args(["run", str(tmp_path), "--arm", "atlas", "--out", "o",
                              "--image", "i", "--compose-project", "atlas",
                              "--workspace-root", str(tmp_path), "--commit", "b0e6013"])
    _, problems = driver.stack_identity(args)
    assert problems and "development stack" in problems[0]


def test_a_run_records_each_session_and_its_grade(tmp_path, monkeypatch):
    suite = make_suite(tmp_path / "s")
    out = tmp_path / "records.jsonl"
    monkeypatch.setattr(driver, "stack_identity", lambda args: ({"commit": "b0e6013"}, []))
    monkeypatch.setattr(driver, "run_atlas",
                        lambda *a, **k: ArmResult("completed", "deliverables_demonstrated", 9.0, 3, 500))
    monkeypatch.setattr(driver, "grade", lambda task, ws, image: G.Grade(G.PASS, "ok"))
    code = driver.main(["run", str(suite), "--arm", "atlas", "--out", str(out), "--image", "img",
                        "--compose-project", "atlaseval", "--workspace-root", str(tmp_path / "w"),
                        "--repeats", "2", "--commit", "b0e6013"])
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert code == 0 and len(recs) == 2
    assert recs[0]["grade"] == G.PASS and recs[0]["stack"] == {"commit": "b0e6013"}
    assert (tmp_path / "w").is_dir() and len(list((tmp_path / "w").iterdir())) == 2


def test_an_unknown_task_id_is_refused(tmp_path):
    tasks = S.load_suite(make_suite(tmp_path))
    with pytest.raises(S.SuiteError, match="not in the suite: t99"):
        driver.select(tasks, "t01,t99")
