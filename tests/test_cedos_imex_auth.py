"""The folder companion sends its access token only to safe destinations."""

import pytest

from imex import cedos_imex


class EmptyResponse:
    status = 204

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return b""


def test_hosted_request_sends_bearer_token(monkeypatch):
    captured = []

    def fake_urlopen(request, timeout):
        captured.append((request, timeout))
        return EmptyResponse()

    monkeypatch.setattr(cedos_imex.urllib.request, "urlopen", fake_urlopen)
    status, body = cedos_imex.api_request(
        "https://realgcp-cedos.hf.space", "GET", "/auto/proposal", token="private-token"
    )
    assert (status, body) == (204, None)
    assert captured[0][0].get_header("Authorization") == "Bearer private-token"


def test_token_refuses_remote_plain_http(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("A request must not be made over plain HTTP")

    monkeypatch.setattr(cedos_imex.urllib.request, "urlopen", fail_if_called)
    with pytest.raises(ValueError, match="HTTPS"):
        cedos_imex.api_request(
            "http://example.com", "GET", "/auto/proposal", token="private-token"
        )
