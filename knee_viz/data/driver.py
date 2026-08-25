"""The Qt boundary: a timer that polls a :class:`DataSource` and emits frames."""

from __future__ import annotations

from PyQt6 import QtCore

from knee_viz.data.source import DataSource, Frame


class FrameDriver(QtCore.QObject):
    """Polls a data source on a timer and emits each new frame."""

    frameReady = QtCore.pyqtSignal(object)
    errorRaised = QtCore.pyqtSignal(str)

    def __init__(self, source: DataSource, interval_ms: int = 20, parent=None) -> None:
        super().__init__(parent)
        self.source = source
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)
        self._last_error: str | None = None

    @property
    def is_running(self) -> bool:
        return self._timer.isActive()

    def start(self) -> None:
        self.source.open()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self.source.close()

    def _tick(self) -> None:
        try:
            frame = self.source.poll()
        except Exception as exc:  # a source must not take the GUI down
            self._report(f"{type(exc).__name__}: {exc}")
            return
        if frame is not None:
            self.frameReady.emit(frame)

    def _report(self, message: str) -> None:
        """Emit an error once per distinct message rather than at the poll rate."""
        if message != self._last_error:
            self._last_error = message
            self.errorRaised.emit(message)
