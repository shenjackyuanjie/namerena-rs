from typing import Tuple, Optional

# from pyglet.gl import gl
from pyglet.math import Mat4, Vec3
from pyglet.graphics import Group


class GroupCamera(Group):
    """
    A camera by group
    can be used by just added to your widget
    """

    def __init__(
        self,
        window,
        order: int = 0,
        parent: Optional[Group] = None,
        view_x: Optional[int] = 0,
        view_y: Optional[int] = 0,
        zoom: Optional[float] = 1.0,
        min_zoom: Optional[float] = 1.0,
        max_zoom: Optional[float] = 1.0,
    ):
        super().__init__(order=order, parent=parent)
        self._window = window
        self._previous_view = None

        self._view_x = view_x or 0
        self._view_y = view_y or 0
        self._zoom = zoom or 1.0
        self.min_zoom = min_zoom or 1.0
        self.max_zoom = max_zoom or 1.0

    @property
    def view_x(self) -> int:
        return self._view_x

    @view_x.setter
    def view_x(self, value: int):
        self._view_x = value

    @property
    def view_y(self) -> int:
        return self._view_y

    @view_y.setter
    def view_y(self, value: int):
        self._view_y = value

    @property
    def zoom(self) -> float:
        return min(max(self._zoom, self.min_zoom), self.max_zoom)

    @zoom.setter
    def zoom(self, value: float):
        self._zoom = value

    def reset(self):
        self._view_x = 0
        self._view_y = 0
        self.zoom = 1

    def set_state(self):
        self._previous_view = self._window.view

        view = Mat4.from_translation(Vec3(self._view_x, self._view_y, 0))
        if self._zoom == 1.0:
            self._window.view = view
        else:
            view = view.scale(Vec3(self._zoom, self._zoom, 1))
            self._window.view = view

    def unset_state(self):
        self._window.view = self._previous_view
