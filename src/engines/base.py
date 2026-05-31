class BaseEngine:
    def process(self, params: dict) -> dict:
        raise NotImplementedError

    def is_available(self) -> bool:
        return True
