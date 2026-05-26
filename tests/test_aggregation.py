from src.aggregation import majority_vote


def test_majority_vote():
    predictions = [
        {"query_id": "q1", "client_id": "c1", "predicted_answer": "B"},
        {"query_id": "q1", "client_id": "c2", "predicted_answer": "B"},
        {"query_id": "q1", "client_id": "c3", "predicted_answer": "C"},
    ]
    final, counts = majority_vote(predictions)
    assert final["q1"] == "B"
    assert counts["q1"]["B"] == 2


def test_tie_breaking_uses_abcd_order():
    predictions = [
        {"query_id": "q1", "client_id": "c1", "predicted_answer": "C"},
        {"query_id": "q1", "client_id": "c2", "predicted_answer": "A"},
    ]
    final, _ = majority_vote(predictions)
    assert final["q1"] == "A"
