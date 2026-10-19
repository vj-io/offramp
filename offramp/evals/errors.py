"""Errors that end an evaluation without producing a measured result."""


class ProviderError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class MissingReference(ValueError):
    def __init__(self, case_ids: list[str]):
        super().__init__(f"{len(case_ids)} case(s) have no reference: {', '.join(case_ids)}")
        self.case_ids = case_ids
