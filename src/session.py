"""
Повторное отправление запросов в случае ошибки
"""

import requests
from requests.sessions import Session as BaseSession
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util.retry import Retry


class TimeoutHTTPAdapter(HTTPAdapter):
    """
    Адаптер, устанавливающий таймаут для всех запросов.
    """

    def __init__(self, *args, timeout: float = 5.0, **kwargs):
        self.timeout = timeout
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        kwargs.setdefault("timeout", self.timeout)
        return super().send(request, **kwargs)


class Session:
    """
    Сессия с поддержкой экспоненциальной задержки при повторных запросах.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 5.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ):
        self.base_url = base_url
        self.timeout = timeout

        self.session = BaseSession()

        # Политика retry
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )

        adapter = TimeoutHTTPAdapter(max_retries=retry, timeout=self.timeout)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get(self, url: str, **kwargs) -> requests.Response:
        return self._safe_request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        return self._safe_request("POST", url, **kwargs)

    def _safe_request(self, method: str, url: str, **kwargs) -> requests.Response:
        full_url = f"{self.base_url}{url}"

        try:
            response = self.session.request(method, full_url, **kwargs)
            response.raise_for_status()
            return response
        except RequestException as e:
            raise e


class RetryError(Exception):
    pass
