import pytest
from service.services.pairs_service import get_pair_result
from service.exceptions import PairNotFoundError, ResultsFileNotFoundError

def test_get_pair_result_return_matches_for_known_pair():
    result = get_pair_result("CVX:DVN", fold=None, static=True)
    assert result["pair_name"] == "CVX:DVN"
    assert len(result["results"]) > 0

def test_get_pair_results_raises_for_unknown_pair():
    with pytest.raises(PairNotFoundError):
        get_pair_result("ZZZ:ZZZ", fold=None, static=True)

def test_get_pair_results_raises_for_missing_file(monkeypatch):
    import service.services.pairs_service as svc  
    monkeypatch.setattr(
        svc,
        "STATIC_FOLDS_CSV", 
        svc.DATA_DIR / "does_not_exist.csv"
    )
    with pytest.raises(ResultsFileNotFoundError):
        get_pair_result("CVX:XOM", fold=None, static=True)
    