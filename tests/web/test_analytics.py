"""Synthetic evidence only: no internet, checkpoints, training or real experiment mutation."""
import hashlib
import json
from pathlib import Path

import pytest

from transformer.web.application import load_analytics, parse_log
from transformer.web.application import create_app
from tests.web.test_application_shell import server_for, request


@pytest.fixture
def evidence(tmp_path):
    def put(path, data, raw=False):
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data if raw else json.dumps(data))
        return {"path": path, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    architecture = dict(vocab_size=4, num_layers=1, num_heads=1, embedding_dim=4,
                        context_length=8, feed_forward_expansion=2, dropout=0.0,
                        initialization_std=0.02, layer_norm_epsilon=1e-5,
                        tie_embeddings=True, schema_version=1)
    tokenizer = put("data/tokens/fixture/tokenizer.json",
                    {"kind":"character","fingerprint":"a"*64,"characters":["a","b"],
                     "special_tokens":{"unk":0,"eos":1}})
    sources = [{"name":"synthetic","source_id":"test/fixture","revision":"fixture","license":"test-only"}]
    put("data/tokens/fixture/manifest.json",
        {"tokenizer_fingerprint":"a"*64,"dataset_fingerprint":"b"*64,"token_data_fingerprint":"c"*64,
         "prepared_sources":sources,"boundary_policy":"packed_with_eos_within_one_stream",
         "vocabulary_fit":{"policy":"Synthetic training characters only"}})
    models, runs, scores, summaries = [], {}, {}, {}
    for kind,member in [("baseline","model_a"),("combined","model_b")]:
        run_id = "fixture-" + kind
        config = {"model":architecture,"run_id":run_id,"token_data_dir":"data/tokens/fixture",
                  "tokenizer_fingerprint":"a"*64,"dataset_fingerprint":"b"*64,
                  "token_data_fingerprint":"c"*64,"max_training_tokens":16}
        base = f"runs/{run_id}/"
        cref = put(base+"configuration.json",config)
        log = "".join(json.dumps({"event":"update","optimizer_update":u,"tokens_processed":u*8,
                                 "training_loss":1/u,"validation_loss":None})+"\n" for u in [1,2])
        lref = put(base+"metrics.jsonl",log,raw=True)
        history = [{"optimizer_update":2,"tokens_processed":16,"validation_loss":0.8}]
        href = put(base+"validation_history.json",{"history":history})
        sref = put(base+"training_summary.json",{"training_session_wall_seconds":2.0})
        runs[member] = {"configuration":config,"run_id":run_id,"checkpoint":{"sha256":"d"*64},
            "run_files_before":{"metrics.jsonl":lref["sha256"],"validation_history.json":href["sha256"],
                               "training_summary.json":sref["sha256"]},
            "validation_history":history,"training_session_wall_seconds":2.0,
            "training_timing_definition":"Synthetic timing only","prepared_sources":sources,
            "parameter_count":236,"selected_training_tokens":16,"selection_rule":"lowest validation loss",
            "training_exposure":{"source_target_tokens":{"synthetic":16}}}
        scores[member]={"packed":{"sources":{"synthetic":{"cross_entropy_nats":0.0,"perplexity":1.0,"target_tokens":8}}}}
        summaries[member]={"pooled_tokens_per_second":4.0,"new_tokens":8,"elapsed_seconds":2.0}
        models.append({"model_id":run_id,"kind":kind,"comparison_member":member,"run_id":run_id,
            "checkpoint":{"path":base+"checkpoints/test.pt","sha256":"d"*64},"configuration":cref,
            "tokenizer":tokenizer,"tokenizer_fingerprint":"a"*64,"checkpoint_update":2})
    evaluation={"runs":runs,"scores":scores,"identity":{"tokenizer_fingerprint":"a"*64},
                "status":"completed","test_results_used_for_selection":False,
                "environment":{"device":"cpu","machine":"synthetic","torch":"fixture"},
                "generation_benchmark":{"definition":"Synthetic test only","summaries":summaries}}
    ref=put("artifacts/fixture/results.json",evaluation)
    for m in models:m["metrics"]=ref
    put("configs/inference/model_registry.json",{"schema_version":1,"models":models})
    return tmp_path, put


def test_values_map_exactly_to_synthetic_sources(evidence):
    root,_=evidence
    data=load_analytics(root)
    assert data["compatible"]
    for m in data["models"]:
        assert m["architecture"]["embedding_dim"]==4
        assert m["training_tokens"]==16
        assert m["training_seconds"]==2.0
        assert m["generation_tokens_per_second"]==4.0
        assert m["test_sources"]["synthetic"]=={"loss":0.0,"perplexity":1.0,"target_tokens":8}
        assert [p["tokens"] for p in m["training"]]==[8,16]
        assert m["validation"][0]["loss"]==0.8
        assert not m["warnings"]
    json.dumps(data,allow_nan=False)


def test_missing_registry_and_no_arbitrary_paths(tmp_path):
    assert not load_analytics(tmp_path)["models"]
    with server_for(create_app(project_root=tmp_path,device="cpu")) as url:
        status,_,raw=request(url,"/api/v1/analytics?path=/etc/passwd")
        assert status==200
        assert json.loads(raw)["models"]==[]
        assert b"root:" not in raw


@pytest.mark.parametrize("content",["",'{bad\n', '{"event":"update","optimizer_update":1,"tokens_processed":8,"training_loss":NaN}\n'])
def test_empty_or_malformed_log_remains_explicit(evidence,content):
    root,put=evidence
    put("runs/fixture-baseline/metrics.jsonl",content,raw=True)
    m=load_analytics(root)["models"][0]
    assert m["status"]=="available"
    assert m["training"]==[]
    assert m["warnings"]


def test_partial_log_retains_valid_rows_without_fabricated_values(evidence):
    root,put=evidence
    put("runs/fixture-baseline/metrics.jsonl",
        '{"event":"update","optimizer_update":1,"tokens_processed":8,"training_loss":0.75}\n{"event":',raw=True)
    m=load_analytics(root)["models"][0]
    assert m["training"][0]["loss"]==0.75
    assert len(m["training"])==1
    assert any("partial" in w for w in m["warnings"])
    assert any("line 2" in w for w in m["warnings"])


def test_missing_log_and_summary_do_not_become_zero(evidence):
    root,_=evidence
    (root/"runs/fixture-baseline/metrics.jsonl").unlink()
    (root/"runs/fixture-baseline/training_summary.json").unlink()
    m=load_analytics(root)["models"][0]
    assert m["training"]==[]
    assert m["training_seconds"] is None
    assert len(m["warnings"])==2


def test_corrupt_registered_evaluation_is_withheld(evidence):
    root,put=evidence
    put("artifacts/fixture/results.json",{"invented":"metrics"})
    data=load_analytics(root)
    assert not data["compatible"]
    assert all(m["status"]=="unavailable" for m in data["models"])


def test_mismatched_configuration_is_withheld(evidence):
    root,put=evidence
    put("runs/fixture-baseline/configuration.json",{"model":{"num_layers":99}})
    data=load_analytics(root)
    assert not data["compatible"]
    assert data["models"][0]["status"]=="unavailable"
    assert data["models"][1]["status"]=="available"


def test_bad_validation_and_manifest_do_not_poison_other_metrics(evidence):
    root,put=evidence
    put("runs/fixture-baseline/validation_history.json",{"history":[{"validation_loss":None}]})
    put("data/tokens/fixture/manifest.json",{"dataset_fingerprint":"wrong"})
    m=load_analytics(root)["models"][0]
    assert m["validation"]==[]
    assert m["manifest"] is None
    assert m["training_seconds"]==2.0


def test_duplicate_nonfinite_and_out_of_order_log_rows_are_rejected():
    rows=[{"event":"update","optimizer_update":u,"tokens_processed":t,"training_loss":loss}
          for u,t,loss in [(1,8,2),(1,8,1),(2,16,float("inf")),(3,24,1),(2,16,1)]]
    points,warnings=parse_log("\n".join(json.dumps(r) for r in rows))
    assert [r["update"] for r in points]==[1,3]
    assert points[-1]["gap_before"]
    assert len(warnings)==4


def test_malformed_but_rehashed_evaluation_has_explicit_unavailable_state(evidence):
    root,put=evidence
    ref=put("artifacts/fixture/results.json",{"runs":None})
    catalog=json.loads((root/"configs/inference/model_registry.json").read_text())
    for row in catalog["models"]:row["metrics"]=ref
    put("configs/inference/model_registry.json",catalog)
    result=load_analytics(root)
    assert not result["compatible"]
    assert all(m["status"]=="unavailable" for m in result["models"])


def test_extreme_integer_log_loss_rejects_only_bad_row():
    bad = {"event": "update", "optimizer_update": 1, "tokens_processed": 8,
           "training_loss": 10 ** 400}
    good = {"event": "update", "optimizer_update": 2, "tokens_processed": 16,
            "training_loss": 1.0}
    points, warnings = parse_log(json.dumps(bad) + "\n" + json.dumps(good))
    assert [p["update"] for p in points] == [2]
    assert any("line 1" in warning for warning in warnings)


def test_exact_log_limit_does_not_claim_omitted_records():
    rows = [json.dumps({"event": "update", "optimizer_update": n,
                        "tokens_processed": n * 8, "training_loss": 1.0})
            for n in range(1, 10002)]
    points, warnings = parse_log("\n".join(rows[:10000]))
    assert len(points) == 10000 and not warnings
    points, warnings = parse_log("\n".join(rows))
    assert len(points) == 10000
    assert any("omitted" in warning for warning in warnings)


def test_shared_registered_json_is_read_once_but_revalidated_next_request(evidence, monkeypatch):
    import transformer.web.application as analytics
    root, put = evidence
    original = analytics._read
    reads = []
    def counted(registry, path, *args, **kwargs):
        reads.append(path)
        return original(registry, path, *args, **kwargs)
    monkeypatch.setattr(analytics, "_read", counted)
    first = load_analytics(root)
    assert first["compatible"]
    assert reads.count("artifacts/fixture/results.json") == 1
    assert reads.count("data/tokens/fixture/tokenizer.json") == 1
    first["models"][0]["architecture"]["num_layers"] = 999
    assert load_analytics(root)["models"][0]["architecture"]["num_layers"] == 1
    put("artifacts/fixture/results.json", {"corrupt": True})
    assert not load_analytics(root)["compatible"]
