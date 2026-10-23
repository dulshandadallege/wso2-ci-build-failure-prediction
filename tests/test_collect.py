from src.collect import file_kind, month_windows, run_row
from src.github_client import hash_id, load_env


def test_file_kind():
    assert file_kind("pom.xml") == "dep"
    assert file_kind("components/core/src/test/java/FooTest.java") == "test"
    assert file_kind(".github/workflows/pr-builder.yml") == "ci"
    assert file_kind("docs/README.md") == "doc"
    assert file_kind("conf/deployment.toml") == "config"
    assert file_kind("ballerina/http_client.bal") == "code"
    assert file_kind("assets/logo.png") == "other"


def test_month_windows():
    windows = list(month_windows(3))
    assert len(windows) == 3
    assert all(".." in w and w.endswith(w[-2:]) for w in windows)
    assert windows[0] > windows[1] > windows[2]


def test_hash_is_salted_and_stable():
    assert hash_id("someone", "s1") == hash_id("Someone", "s1")
    assert hash_id("someone", "s1") != hash_id("someone", "s2")
    assert len(hash_id("someone", "s1")) == 12
    assert hash_id("", "s1") == ""


def test_run_row_does_not_store_raw_names():
    run = {
        "id": 1, "name": "PR Build", "run_number": 5, "run_attempt": 1, "event": "pull_request",
        "head_branch": "feature/secret-branch", "head_sha": "abc", "display_title": "Fix",
        "actor": {"login": "real-user"}, "triggering_actor": {"login": "real-user"},
        "head_repository": {"full_name": "real-user/repo"}, "created_at": "2026-01-01T00:00:00Z",
        "run_started_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:10:00Z", "conclusion": "failure",
    }
    row = run_row(run, "org/repo", "salt")
    values = " ".join(str(v) for v in row.values())
    assert "real-user" not in values
    assert "secret-branch" not in values
    assert row["is_fork"] == 1


def test_load_env_reads_file(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text('GITHUB_TOKEN="abc123"\n# comment\nHASH_SALT=xyz\n')
    env = load_env(env_file)
    assert env["GITHUB_TOKEN"] == "abc123"
    assert env["HASH_SALT"] == "xyz"
