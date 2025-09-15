class OptimizationError(Exception):
    """
    When an optimization routine fails, where cvxpy has not returned the 'optimal' flag
    """

    def __init__(self, *args, **kwargs):
        default_message = (
            "Please check your objectives/constraints or use a different solver."
        )
        super().__init__(default_message, *args, **kwargs)