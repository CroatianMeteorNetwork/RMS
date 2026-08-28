from collections import deque
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Detection:
    timestamp: float
    data: dict
    filename: str


class FalsePositiveFilter:
    """
    Buffers detections in a sliding time window. A detection is only
    confirmed once it ages out of the window AND the
    window it lived in never exceeded max_detections.
    Drawback is that detections won't be sent to the server as they are found,
    but will have to be held for the duration of the time window before they are sent causing delay.
    This fitler relies on files being processed chronologically in the order of their recording.
    """

    def __init__(
        self,
        window_seconds: float,
        max_detections: int,
        on_confirmed: Callable[[Detection], None],
        log,
    ):
        self.window_seconds = window_seconds
        self.max_detections = max_detections
        self.on_confirmed = on_confirmed
        self._buffer: deque[Detection] = deque()
        self.log = log

    def add_detection(self, timestamp: float, data: dict, filename: str) -> None:
        self._buffer.append(Detection(timestamp, data, filename))
        self._process(timestamp)

    def tick(self, now: float) -> None:
        """Call periodically (e.g. every inference cycle) even when
        there's no new detection, so aged-out detections still get
        resolved."""
        self.log.debug("Filter tick.")
        self._process(now)

    def _process(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._buffer and self._buffer[0].timestamp < cutoff:
            expired = self._buffer.popleft()
            # window state at the moment this detection ages out,
            # AFTER removing it but with the rest of the window intact
            if len(self._buffer) + 1 <= self.max_detections:
                self.log.info(f"{expired.filename} passed the filter. Uploading...")
                self.on_confirmed(expired)
            else:
                self.log.info(f"{expired.filename} didn't pass the filter.")
