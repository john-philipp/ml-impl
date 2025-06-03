import time


class Stopwatch:
    def __init__(self):
        self._times = []
        self._running = False

    def start(self):
        if self._running:
            raise AssertionError("Stopwatch already running.")
        self._times.append(self._get_time())
        self._running = True

    def lap(self):
        if not self._running:
            raise AssertionError("Stopwatch not yet running.")
        self._times.append(self._get_time())
        return self.elapsed_last_lap()

    def stop(self):
        if not self._running:
            raise AssertionError("Stopwatch not yet running.")
        self.lap()
        self._running = False
        return self.elapsed_total()

    def reset(self):
        self._times.clear()
        self._running = False

    def elapsed_last_lap(self):
        if len(self._times) < 2:
            raise ValueError("Need at least one complete lap first.")
        return self._times[-1] - self._times[-2]

    def elapsed_total(self):
        if len(self._times) < 2:
            raise ValueError("Need at least one complete lap first.")
        return self._times[-1] - self._times[0]

    @staticmethod
    def _get_time():
        return time.perf_counter()
