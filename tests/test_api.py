import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from app.main import app  # noqa: E402

client = fastapi_testclient.TestClient(app)


class TestHealth:
    def test_healthz_reports_loaded_agents_and_cases(self):
        response = client.get("/healthz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "v2" in body["agents"]
        assert body["cases"] > 0

    def test_agents_are_listed(self):
        assert set(client.get("/agents").json()["agents"]) == {"v1", "v2"}


class TestRunEndpoint:
    def test_running_the_current_agent_passes_the_gate(self):
        response = client.post(
            "/evals/run",
            json={"agent": "v2", "judge": "rubric", "compare_to_baseline": False},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["gate"]["passed"], body["gate"]["violations"]
        assert body["n_cases"] > 0
        assert body["run_id"]

    def test_running_the_known_bad_agent_fails_the_gate(self):
        response = client.post(
            "/evals/run",
            json={"agent": "v1", "judge": "rubric", "compare_to_baseline": False},
        )
        body = response.json()
        assert not body["gate"]["passed"]
        assert body["gate"]["violations"]

    def test_unknown_agent_returns_404(self):
        response = client.post("/evals/run", json={"agent": "v99", "judge": "rubric"})
        assert response.status_code == 404

    def test_case_level_results_are_opt_in(self):
        summary = client.post(
            "/evals/run",
            json={"agent": "v2", "judge": "rubric", "compare_to_baseline": False},
        ).json()
        stored = client.get(f"/evals/{summary['run_id']}").json()
        assert "results" not in stored

        detailed = client.post(
            "/evals/run",
            json={
                "agent": "v2",
                "judge": "rubric",
                "compare_to_baseline": False,
                "include_cases": True,
            },
        ).json()
        stored = client.get(f"/evals/{detailed['run_id']}").json()
        assert len(stored["results"]) == stored["n_cases"]

    def test_unknown_run_returns_404(self):
        assert client.get("/evals/does-not-exist").status_code == 404

    def test_runs_are_listed(self):
        client.post(
            "/evals/run",
            json={"agent": "v2", "judge": "rubric", "compare_to_baseline": False},
        )
        assert client.get("/evals").json()["runs"]


class TestJudgeAuditEndpoint:
    def test_audit_reports_judge_accuracy(self):
        body = client.get("/judge/audit", params={"name": "rubric"}).json()
        assert body["judge"] == "rubric"
        assert 0.0 <= body["accuracy"] <= 1.0
        assert body["trustworthy"] is True


class TestMetricsEndpoint:
    def test_prometheus_exposition_format(self):
        client.post(
            "/evals/run",
            json={"agent": "v2", "judge": "rubric", "compare_to_baseline": False},
        )
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        body = response.text
        assert "# TYPE agent_eval_pass_rate gauge" in body
        assert "agent_eval_gate_passed" in body
