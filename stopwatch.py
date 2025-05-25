import time


class Stopwatch:
    def __init__(self):
        self._start = None
        self._elapsed = 0
        self._running = False

    def start(self):
        if not self._running:
            self._start = time.perf_counter()
            self._running = True

    def stop(self):
        if self._running:
            self._elapsed += time.perf_counter() - self._start
            self._running = False

    def reset(self):
        self._start = None
        self._elapsed = 0
        self._running = False

    def elapsed(self):
        if self._running:
            return self._elapsed + (time.perf_counter() - self._start)
        return self._elapsed
