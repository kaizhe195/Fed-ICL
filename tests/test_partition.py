from src.partition import dirichlet_partition


def _examples():
    return [
        {"id": f"ex_{i}", "subject": "math" if i < 5 else "history", "answer": "A"}
        for i in range(10)
    ]


def test_all_examples_are_assigned_to_clients():
    clients = dirichlet_partition(_examples(), num_clients=3, alpha=10.0, seed=1)
    assigned = [example["id"] for rows in clients.values() for example in rows]
    assert sorted(assigned) == [f"ex_{i}" for i in range(10)]


def test_no_example_appears_twice():
    clients = dirichlet_partition(_examples(), num_clients=3, alpha=10.0, seed=1)
    assigned = [example["id"] for rows in clients.values() for example in rows]
    assert len(assigned) == len(set(assigned))


def test_num_clients_is_respected():
    clients = dirichlet_partition(_examples(), num_clients=4, alpha=10.0, seed=1)
    assert sorted(clients.keys()) == ["client_0", "client_1", "client_2", "client_3"]
