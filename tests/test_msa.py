import io
import json
import tarfile

import pytest

from abcfold.msa.cache import (MsaCache, msa_cache_key, normalize_sequence,
                               paired_cache_key)
from abcfold.msa.colabfold import (ColabFoldClient, ColabFoldError,
                                   merge_query_blocks, split_colabfold_a3m)
from abcfold.msa.export import export_msas
from abcfold.msa.formats import a3m_to_boltz_csv, chai_msa_frame, parse_a3m
from abcfold.msa.resolve import MsaOptions, parse_chain_paths, resolve_msas
from abcfold.msa.stats import MAX_BINS, bin_values, msa_chain_stats

SEQ_A = "MKTAYIAKQR"
SEQ_B = "GSHMLEDPV"


def a3m(query, n, prefix="UniRef100_X"):
    rows = [f">query\n{query}\n"]
    for i in range(n):
        rows.append(f">{prefix}{i}\n{query[:-1]}{'ACDEFGHIKL'[i % 10]}\n")
    return "".join(rows)


# --- cache hashing -------------------------------------------------------


def test_cache_key_normalises_sequence():
    assert normalize_sequence(" mkt ay\n") == "MKTAY"
    assert msa_cache_key("mktay") == msa_cache_key("MKTAY ")


def test_cache_key_depends_on_source_and_options():
    keys = {
        msa_cache_key(SEQ_A, "server", ""),
        msa_cache_key(SEQ_A, "custom", ""),
        msa_cache_key(SEQ_A, "server", "env=0"),
        msa_cache_key(SEQ_B, "server", ""),
    }
    assert len(keys) == 4
    assert len(msa_cache_key(SEQ_A)) == 64


def test_paired_key_depends_on_partners_and_index():
    k0 = paired_cache_key([SEQ_A, SEQ_B], 0)
    assert k0 != paired_cache_key([SEQ_A, SEQ_B], 1)
    assert k0 != paired_cache_key([SEQ_A, "MMMM"], 0)
    assert k0 != msa_cache_key(SEQ_A)


def test_cache_roundtrip(tmp_path):
    cache = MsaCache(tmp_path)
    key = msa_cache_key(SEQ_A)
    assert cache.get(key) is None
    path = cache.put(key, "a3m", {"source": "server"})
    assert path == tmp_path / key[:2] / key / "msa.a3m"
    assert cache.get(key) == "a3m"
    assert cache.meta(key)["source"] == "server"


# --- ColabFold client ----------------------------------------------------


class FakeResponse:
    def __init__(self, payload=None, content=b""):
        self.payload = payload
        self.content = content
        self.text = json.dumps(payload)

    def json(self):
        return self.payload


def make_tar(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class FakeSession:
    def __init__(self, files, statuses=("RUNNING", "COMPLETE")):
        self.files = files
        self.statuses = list(statuses)
        self.posts = []

    def post(self, url, data, headers, timeout):
        self.posts.append((url, data))
        return FakeResponse({"id": "T1", "status": "PENDING"})

    def get(self, url, headers, timeout):
        if "/ticket/" in url:
            return FakeResponse({"status": self.statuses.pop(0)})
        return FakeResponse(content=make_tar(self.files))


def colabfold_block(qid, query, hits):
    lines = [f">{qid}\n{query}\n"] + [f">hit{qid}_{i}\n{h}\n" for i, h in
                                       enumerate(hits)]
    return "".join(lines)


def test_split_and_merge_colabfold_blocks():
    text = (colabfold_block(101, SEQ_A, ["MKTAYIAKQA"]) + "\x00"
            + colabfold_block(102, SEQ_B, ["GSHMLEDPA"]))
    blocks = split_colabfold_a3m(text)
    assert set(blocks) == {101, 102}
    env = colabfold_block(101, SEQ_A, ["MKTAYIAKQE"])
    merged = parse_a3m(merge_query_blocks([blocks[101], env], 101))
    assert merged[0] == ("query", SEQ_A)
    assert [s for _, s in merged] == [SEQ_A, "MKTAYIAKQA", "MKTAYIAKQE"]


def test_client_unpaired_maps_queries_back():
    uniref = (colabfold_block(101, SEQ_A, ["MKTAYIAKQA"]) + "\x00"
              + colabfold_block(102, SEQ_B, ["GSHMLEDPA"]))
    env = colabfold_block(101, SEQ_A, ["MKTAYIAKQE"]) + "\x00" + colabfold_block(
        102, SEQ_B, [])
    session = FakeSession({
        "uniref.a3m": uniref,
        "bfd.mgnify30.metaeuk30.smag30.a3m": env,
    })
    client = ColabFoldClient("http://msa.test/", session=session,
                             sleep=lambda s: None)
    # Server numbers unique queries from 101: SEQ_A -> 101, SEQ_B -> 102.
    out = client.unpaired([SEQ_A, SEQ_B, SEQ_A])
    # duplicates are submitted once
    assert session.posts[0][1]["q"].count(">") == 2
    assert session.posts[0][0] == "http://msa.test/ticket/msa"
    assert session.posts[0][1]["mode"] == "env"
    assert parse_a3m(out[0])[0][1] == SEQ_A
    assert len(parse_a3m(out[0])) == 3
    assert parse_a3m(out[1])[0][1] == SEQ_B
    assert out[0] == out[2]


def test_client_raises_on_error_status():
    session = FakeSession({}, statuses=["ERROR"])
    client = ColabFoldClient(session=session, sleep=lambda s: None)
    with pytest.raises(ColabFoldError):
        client.unpaired([SEQ_A])


# --- formats and stats ---------------------------------------------------


def test_boltz_csv_pairs_rows_by_key():
    paired = f">query\n{SEQ_A}\n>p1\nMKTAYIAKQA\n>p2\n{'-' * len(SEQ_A)}\n"
    csv = a3m_to_boltz_csv(a3m(SEQ_A, 3), paired).splitlines()
    assert csv[0] == "key,sequence"
    assert csv[1] == f"0,{SEQ_A}"
    assert csv[2] == "1,MKTAYIAKQA"
    # gap-only paired row dropped, unpaired query not duplicated
    assert [r.split(",")[0] for r in csv[3:]] == ["-1"] * 3


def test_chai_frame_columns_and_pairing():
    paired = f">query\n{SEQ_A}\n>p1\nMKTAYIAKQA\n"
    frame = chai_msa_frame(a3m(SEQ_A, 2), paired)
    assert list(frame.columns) == ["sequence", "source_database", "pairing_key",
                                   "comment"]
    assert frame.iloc[0]["source_database"] == "query"
    assert list(frame["pairing_key"][:2]) == ["0", "1"]
    assert frame["pairing_key"].iloc[-1] == ""


def test_bin_values_limits_to_max_bins():
    assert bin_values([1, 2, 3]) == [1.0, 2.0, 3.0]
    binned = bin_values(list(range(1000)))
    assert len(binned) == MAX_BINS
    assert binned[0] == pytest.approx(2.0)


def test_msa_chain_stats():
    stats = msa_chain_stats(a3m(SEQ_A, 4), None, len(SEQ_A), "server", 1.5, True)
    assert stats["depth"] == 5
    assert stats["paired_depth"] == 0
    assert stats["source"] == "server"
    assert stats["cached"] is True
    assert len(stats["coverage"]) == len(SEQ_A)
    assert max(stats["coverage"]) == 5


# --- resolver ------------------------------------------------------------


def complex_input(*seqs):
    return {
        "name": "t",
        "modelSeeds": [1],
        "sequences": [
            {"protein": {"id": chr(65 + i), "sequence": s}} for i, s in enumerate(seqs)
        ] + [{"ligand": {"id": "Z", "ccdCodes": ["ATP"]}}],
    }


class FakeClient:
    def __init__(self):
        self.unpaired_calls = []
        self.paired_calls = []

    def unpaired(self, seqs, use_env=True):
        self.unpaired_calls.append(list(seqs))
        return [a3m(s, 12) for s in seqs]

    def paired(self, seqs, strategy="greedy"):
        self.paired_calls.append(list(seqs))
        return [f">query\n{s}\n>p\n{s}\n" for s in seqs]


def test_resolve_server_caches_and_pairs(tmp_path):
    client = FakeClient()
    opts = MsaOptions(source="server", cache_dir=str(tmp_path))
    params = complex_input(SEQ_A, SEQ_B, SEQ_A)
    result = resolve_msas(params, opts, client=client)
    assert client.unpaired_calls == [[SEQ_A, SEQ_B]]
    assert client.paired_calls == [[SEQ_A, SEQ_B]]
    proteins = [s["protein"] for s in result.params["sequences"] if "protein" in s]
    assert all(p["unpairedMsa"].startswith(">query") for p in proteins)
    assert all(p["pairedMsa"] for p in proteins)
    assert all(p["templates"] == [] for p in proteins)
    assert set(result.chains) == {"A", "B", "C"}
    assert result.chains["A"]["cached"] is False
    # input is not mutated
    assert "unpairedMsa" not in params["sequences"][0]["protein"]

    again = resolve_msas(params, opts, client=FakeClient())
    assert again.chains["A"]["cached"] is True
    assert again.params == result.params


def test_resolve_server_homomer_is_not_paired(tmp_path):
    client = FakeClient()
    result = resolve_msas(complex_input(SEQ_A, SEQ_A),
                          MsaOptions(source="server", cache_dir=str(tmp_path)),
                          client=client)
    assert client.paired_calls == []
    assert result.params["sequences"][0]["protein"]["pairedMsa"] == ""


def test_resolve_none_and_auto():
    none = resolve_msas(complex_input(SEQ_A), MsaOptions(source="none"))
    protein = none.params["sequences"][0]["protein"]
    assert protein["unpairedMsa"] == "" and protein["pairedMsa"] == ""
    auto = resolve_msas(complex_input(SEQ_A), MsaOptions(source="auto"))
    assert auto.source == "backend"
    assert "unpairedMsa" not in auto.params["sequences"][0]["protein"]


def test_resolve_custom(tmp_path):
    path = tmp_path / "a.a3m"
    path.write_text(a3m(SEQ_A, 11))
    path_b = tmp_path / "b.a3m"
    path_b.write_text(a3m(SEQ_B, 3))
    result = resolve_msas(
        complex_input(SEQ_A, SEQ_B),
        MsaOptions(source="custom", custom_a3m=parse_chain_paths([f"A={path}",
                                                                  f"B={path_b}"])),
    )
    assert result.chains["A"]["depth"] == 12
    empty = tmp_path / "empty.a3m"
    empty.write_text("")
    for chains in ([f"A={path}"], [f"A={path}", f"B={path}"], [f"A={empty}"]):
        with pytest.raises(ValueError):
            resolve_msas(complex_input(SEQ_A, SEQ_B),
                         MsaOptions(source="custom",
                                    custom_a3m=parse_chain_paths(chains)))


def test_resolve_af3_data_json(tmp_path):
    data = complex_input(SEQ_A)
    data["sequences"][0]["protein"].update(
        unpairedMsa=a3m(SEQ_A, 10), pairedMsa="", templates=[{"mmcif": "x"}]
    )
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "t_data.json").write_text(json.dumps(data))
    result = resolve_msas(complex_input(SEQ_A),
                          MsaOptions(source="af3", af3_data_json=str(tmp_path)))
    protein = result.params["sequences"][0]["protein"]
    assert protein["templates"] == [{"mmcif": "x"}]
    assert result.chains["A"]["depth"] == 11


def test_parse_chain_paths():
    assert parse_chain_paths(["A,B=x.a3m", "C=y.a3m", "z.a3m"]) == {
        "A": "x.a3m", "B": "x.a3m", "C": "y.a3m", "*": "z.a3m"}


def test_export_formats(tmp_path):
    params = resolve_msas(complex_input(SEQ_A, SEQ_B),
                          MsaOptions(source="server", cache_dir=str(tmp_path / "c")),
                          client=FakeClient()).params
    written = export_msas(params, tmp_path / "out",
                          ["boltz", "chai1", "protenix", "openfold3"])
    assert (tmp_path / "out/boltz/A.csv").exists()
    assert len(written["chai1"]) == 2
    assert (tmp_path / "out/protenix/A/non_pairing.a3m").exists()
    assert (tmp_path / "out/protenix/A/pairing.a3m").exists()
    assert (tmp_path / "out/openfold3/B/colabfold_main.a3m").exists()
