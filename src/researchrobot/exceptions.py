class StopExecution(Exception):
    """For stopping execution of a Jupyter notebook"""

    def _render_traceback_(self):
        pass
