class PairNotFoundError(Exception):
    def __init__(self, pair_name: str):
        self.pair_name = pair_name
        super().__init__(f"No results found for pair '{pair_name}'")

class ResultsFileNotFoundError(Exception):
    def __init__(self, source: str):
        self.source = source
        super().__init__(f"{source} results file not found")