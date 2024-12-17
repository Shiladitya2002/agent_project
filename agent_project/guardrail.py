from abc import ABC, abstractmethod
from overrides import overrides


class BaseGuardrail(ABC):
    @abstractmethod
    def __call__(self, text: str) -> bool:
        """
        detects if input is harmful/inappropriate.
        """
        return False


class SelfIEGuardrail(BaseGuardrail):

    def __init__(self):
        pass

    @overrides
    def __call__(self, text: str) -> str:
        pass


class KeywordFilterGuardrail(BaseGuardrail):

    def __init__(self):
        pass

    @overrides
    def __call__(self, text: str) -> str:
        pass    # TODO:


