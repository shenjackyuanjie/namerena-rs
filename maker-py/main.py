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

    wait = 0

    # 血量
    hp = 1
    # 攻击
    attack = 2
    # 防御
    defense = 3
    # 速度
    speed = 4
    # 敏捷
    agility = 5
    # 魔法
    magic = 6
    # 抗性
    resistance = 7
    # 智慧
    wisdom = 8


class NumWidget:
    def __init__(self, num: int, batch: Batch, group: Group, x: int, y: int) -> None:
        self._y = y
        self._x = x
        font = load_font("黑体", 15)
        font_height = font.ascent - font.descent
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
            group=group,
        )
        self.background = Rectangle(
            x=x,
            y=y,
            width=35,
            height=font_height + 4,
            color=gray,
            batch=batch,
            group=group,
        )

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
        for i in range(256):
            num_name = NumWidget(
                num=i, batch=self.num_batch, group=self.num_group, x=40, y=50
            )
            self.num_dict[i] = num_name
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
            NumStatus.hp: [self.num_dict[i] for i in range(3, 7)],
            NumStatus.attack: [self.num_dict[i] for i in range(10, 13)],
            NumStatus.defense: [self.num_dict[i] for i in range(13, 16)],
            NumStatus.speed: [self.num_dict[i] for i in range(16, 19)],
            NumStatus.agility: [self.num_dict[i] for i in range(19, 22)],
            NumStatus.magic: [self.num_dict[i] for i in range(22, 25)],
            NumStatus.resistance: [self.num_dict[i] for i in range(25, 28)],
            NumStatus.wisdom: [self.num_dict[i] for i in range(28, 31)],
            NumStatus.wait: [
                *(self.num_dict[i] for i in range(0, 3)),
                *(self.num_dict[i] for i in range(7, 10)),
                *(self.num_dict[i] for i in range(31, 256)),
            ],
        }
        self.update_num_display()
    
    def update_num_display(self) -> None:
        
        for status, widgets in self.display_dict.items():
            num_count = 0
            for widget in widgets:
                print(f"status: {status}, num_count: {num_count} {status.value=}")
                widget.x = 40 + (40 * status.value)
                widget.y = self.height - (170 + 30 * num_count)
                num_count += 1
        

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

    def start(self) -> None:
        pyglet.app.run(interval=1 / 30)


if __name__ == "__main__":
    window = MainWindow()
    glClearColor(1, 1, 1, 1)
    window.start()
