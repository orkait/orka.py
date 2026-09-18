import pytest

from orka.autoquant.gratis import GratisClient, GratisError


def test_client_rejects_response_without_tool_action():
    client = GratisClient("http://gratis", "gratis-auto", 1.0, post=lambda *_: {"choices": []})
    with pytest.raises(GratisError, match="tool action"):
        client.next_action([], [])


def test_client_rejects_budget_overrun():
    client = GratisClient("http://gratis", "gratis-auto", 0.01,
                          post=lambda *_: {"usage": {"cost": 0.02}, "choices": []})
    with pytest.raises(GratisError, match="budget"):
        client.next_action([], [])
