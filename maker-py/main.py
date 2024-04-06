import pyglet
from pyglet.font import load as load_font
from pyglet.text import Label
from pyglet.gui import TextEntry
from pyglet.window import Window
from pyglet.gl import glClearColor
from pyglet.shapes import Rectangle
from pyglet.graphics import Batch, Group

from control import RePositionFrame

from enum import IntEnum

gray = (200, 200, 200)


class NumStatus(IntEnum):
    """未被选中"""

    wait = 8

    # 血量
    hp = 0
    # 攻击
    attack = 1
    # 防御
    defense = 2
    # 速度
    speed = 3
    # 敏捷
    agility = 4
    # 魔法
    magic = 5
    # 抗性
    resistance = 6
    # 智慧
    wisdom = 7


class NumWidget:
    def __init__(self, num: int, batch: Batch, group: Group, x: int, y: int) -> None:
        self._y = y
        self._x = x
        font = load_font("黑体", 15)
        font_height = font.ascent - font.descent
        self.label_group = Group(parent=group, order=20)
        self.background_group = Group(parent=group, order=10)
        self.label = Label(
            x=x + 17,
            y=y + 7,
            color=(0, 0, 0, 255),
            text=f"{num}",
            font_name="黑体",
            font_size=15,
            width=35,
            height=font_height + 4,
            anchor_x="center",
            batch=batch,
            group=self.label_group,
        )
        self.background = Rectangle(
            x=x,
            y=y,
            width=35,
            height=font_height + 4,
            color=gray,
            batch=batch,
            group=self.background_group,
        )

    @property
    def value(self) -> int:
        return int(self.label.text)

    @property
    def x(self) -> int:
        return self._x

    @x.setter
    def x(self, value: int) -> None:
        self._x = value
        self.label.x = value + 17
        self.background.x = value

    @property
    def y(self) -> int:
        return self._y

    @y.setter
    def y(self, value: int) -> None:
        self._y = value
        self.label.y = value + 7
        self.background.y = value

    def aabb(self, x: int, y: int) -> bool:
        # 判断是否在范围内
        width = 35
        height = 20
        return self.x <= x <= self.x + width and self.y <= y <= self.y + height


def middle_widget(一: NumWidget, 二: NumWidget, 三: NumWidget) -> int:
    """返回中间值"""
    a, b, c = 一.value, 二.value, 三.value
    if a < b < c or c < b < a:
        return b
    if b < a < c or c < a < b:
        return a
    return c


class MainWindow(Window):
    def __init__(self, *args, **kwargs):
        super().__init__(
            resizable=True,
            width=800,
            height=600,
            caption="Maker",
            vsync=True,
            *args,
            **kwargs,
        )

        self.main_batch = Batch()
        self.main_group = Group()
        self.main_frame = RePositionFrame(self)

        self.name_info_displays = {}
        self.init_name_dispaly()
        self.init_name_diy()

    def init_name_diy(self) -> None:
        """
        初始化 名字自定义
        """
        # 0-255
        self.num_dict = {}
        self.num_batch = Batch()
        self.num_group = Group(parent=self.main_group, order=10)
        # 从大到小
        num_group = Group(parent=self.num_group, order=10)
        for i in range(256):
            num_name = NumWidget(
                num=i, batch=self.num_batch, group=num_group, x=40, y=50
            )
            self.num_dict[i] = num_name
        self.num_hints = []
        # 每个部分的取值提示
        num_hint_group = Group(parent=self.main_group, order=20)
        # hp: 3~6 len = 4
        # 要覆盖住 4 个数字
        self.num_hints.append(
            Rectangle(
                x=40 - 3,
                y=self.height - (170 + 30 * 3),
                width=46,
                height=80,
                color=(255, 255, 255, 255),
                batch=self.num_batch,
                group=num_hint_group,
            )
        )
        
        # 0-9 sorted
        # 取前9个拿到血量这边
        # index 3~6 之和 + 154 = 血量
        # index 10~12 中值 + 36 = 攻击
        # index 13~15 中值 + 36 = 防御
        # index 16~18 中值 + 36 = 速度
        # index 19~21 中值 + 36 = 敏捷
        # index 22~24 中值 + 36 = 魔法
        # index 25~27 中值 + 36 = 抗性
        # index 28~30 中值 + 36 = 智慧
        self.display_dict: dict[NumStatus, list[NumWidget]] = {
            NumStatus.hp: [self.num_dict[i] for i in range(0, 10)],
            NumStatus.attack: [self.num_dict[i] for i in range(10, 13)],
            NumStatus.defense: [self.num_dict[i] for i in range(13, 16)],
            NumStatus.speed: [self.num_dict[i] for i in range(16, 19)],
            NumStatus.agility: [self.num_dict[i] for i in range(19, 22)],
            NumStatus.magic: [self.num_dict[i] for i in range(22, 25)],
            NumStatus.resistance: [self.num_dict[i] for i in range(25, 28)],
            NumStatus.wisdom: [self.num_dict[i] for i in range(28, 31)],
            NumStatus.wait: [self.num_dict[i] for i in range(31, 256)],
        }
        self.update_num_display()

    def update_num_display(self) -> None:
        # sort hp
        self.display_dict[NumStatus.hp].sort(key=lambda x: x.value)
        # sort wait
        self.display_dict[NumStatus.wait].sort(key=lambda x: x.value)

        for status, widgets in self.display_dict.items():
            num_count = 0
            for widget in widgets:
                widget.x = 40 + (65 * status.value)
                widget.y = self.height - (170 + 30 * num_count)
                num_count += 1
        # 计算数据
        hp = sum(widget.value for widget in self.display_dict[NumStatus.hp][3:6]) + 154
        attack = middle_widget(*self.display_dict[NumStatus.attack]) + 36
        defense = middle_widget(*self.display_dict[NumStatus.defense]) + 36
        speed = middle_widget(*self.display_dict[NumStatus.speed]) + 36
        agility = middle_widget(*self.display_dict[NumStatus.agility]) + 36
        magic = middle_widget(*self.display_dict[NumStatus.magic]) + 36
        resistance = middle_widget(*self.display_dict[NumStatus.resistance]) + 36
        wisdom = middle_widget(*self.display_dict[NumStatus.wisdom]) + 36
        gather = sum(
            (int(hp / 3), attack, defense, speed, agility, magic, resistance, wisdom)
        )
        self.name_info_displays[
            "label"
        ].text = f"HP|{hp} 攻|{attack} 防|{defense} 速|{speed} 敏|{agility} 魔|{magic} 抗|{resistance} 智|{wisdom} 八围:{gather}"

    def init_name_dispaly(self) -> None:
        """
        初始化 名字显示 这块内容
        """
        name_group = Group(parent=self.main_group)
        self.name_info_displays["group"] = name_group

        font = load_font("黑体", 20)
        font_height = font.ascent - font.descent
        name_rec = Rectangle(
            x=20,
            y=self.height - 135,
            width=600,  # 在 callback 中定义
            height=font_height,
            # 颜色: 灰色
            color=gray,
            batch=self.main_batch,
            group=name_group,
        )
        name_info_label = Label(
            x=25,
            y=self.height - 127,
            text="HP|{} 攻|{} 防|{} 速|{} 敏|{} 魔|{} 抗|{} 智|{} 八围:{}",
            width=400,
            multiline=False,
            font_name="黑体",
            font_size=15,
            color=(0, 0, 0, 255),
            batch=self.main_batch,
            group=name_group,
        )
        name_entry = TextEntry(
            x=40,
            y=self.height - 100,
            width=200,
            text="x@x",
            # 灰色背景
            color=(*gray, 255),
            text_color=(0, 0, 0, 255),
            batch=self.main_batch,
            group=name_group,
        )

        def rec_callback(rec, width: int, height: int, window: Window):
            # rec.x = 20
            rec.y = height - 135

        self.main_frame.add_callback_func(name_rec, rec_callback)
        self.main_frame.add_calculate_func(
            name_info_label,
            lambda obj, width, height, window: (25, height - 127, 0),
        )
        self.main_frame.add_calculate_func(
            name_entry,
            lambda obj, width, height, window: (40, height - 100),
        )
        self.push_handlers(name_entry)
        self.name_info_displays["rec"] = name_rec
        self.name_info_displays["label"] = name_info_label
        self.name_info_displays["entry"] = name_entry

    def on_draw(self) -> None:
        self.clear()
        self.main_batch.draw()
        self.num_batch.draw()

    def on_resize(self, width, height):
        super().on_resize(width, height)
        self.update_num_display()

    def start(self) -> None:
        pyglet.app.run(interval=1 / 30)


if __name__ == "__main__":
    window = MainWindow()
    glClearColor(1, 1, 1, 1)
    window.start()
