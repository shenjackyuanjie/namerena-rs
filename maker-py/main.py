import pyglet
from pyglet.font import load as load_font
from pyglet.text import Label
from pyglet.gui import TextEntry
from pyglet.window import Window, mouse
from pyglet.gl import glClearColor
from pyglet.shapes import Rectangle
from pyglet.graphics import Batch, Group

from control import RePositionFrame

from typing import List, Dict
from enum import IntEnum

gray = (200, 200, 200)

_version_ = "1.1.0"


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
        self._display = True
        font = load_font("黑体", 13)
        font_height = font.ascent - font.descent
        self.label_group = Group(parent=group, order=20)
        self.background_group = Group(parent=group, order=10)
        self.val = num
        self._value = num & 63
        self.label = Label(
            x=x + 37,
            y=y + 7,
            color=(0, 0, 0, 255),
            text=f"{self._value}-{self.val}",
            font_name="黑体",
            font_size=13,
            width=50,
            height=font_height + 4,
            anchor_x="center",
            batch=batch,
            group=self.label_group,
        )
        self.background = Rectangle(
            x=x,
            y=y,
            width=56,
            height=font_height + 7,
            color=gray,
            batch=batch,
            group=self.background_group,
        )

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, value: int) -> None:
        self._value = value & 63
        self.val = value
        self.label.text = f"{self._value}-{self.val}"

    @property
    def display(self) -> bool:
        return self._display

    @display.setter
    def display(self, value: bool) -> None:
        self._display = value
        self.label_group.visible = value
        self.background_group.visible = value

    @property
    def x(self) -> int:
        return self._x

    @x.setter
    def x(self, value: int) -> None:
        self._x = value
        self.label.x = value + 27
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
        width = self.background.width
        height = self.background.height
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
        self.on_middle = False
        self.middle_base = (0, 0)
        self.drag_speed = 0
        self.drag_start = None
        self.name_data = []

        self.name_info_displays = {}
        self.init_info()
        self.init_name_dispaly()
        self.init_name_diy()

    def init_info(self) -> None:
        """初始化信息显示"""
        self.info_label = Label(
            x=20,
            y=self.height - 50,
            text=f"名字竞技场, 八围制造器v{_version_} by shenjackyuanjie(点完导出看控制台)",
            multiline=False,
            font_name="黑体",
            font_size=17,
            batch=self.main_batch,
            group=Group(parent=self.main_group, order=30),
            color=(0, 0, 0, 255),
        )
        self.output_button = Rectangle(
            x=400,
            y=200,
            width=100,
            height=50,
            color=(0, 0, 255, 200),
            batch=self.main_batch,
            group=self.main_group,
        )
        self.output_label = Label(
            x=400 + 50,
            y=200 + 25,
            text="导出",
            width=100,
            height=50,
            multiline=False,
            anchor_x="center",
            font_name="黑体",
            font_size=20,
            color=(255, 255, 255, 255),
            batch=self.main_batch,
            group=self.main_group,
        )

    def init_name_diy(self) -> None:
        """
        初始化 名字自定义
        """
        # 0-255
        self.num_dict = {}
        self.num_group = Group(parent=self.main_group, order=10)
        # 用于覆盖掉 num 顶上多出来的部分
        cover_group = Group(parent=self.main_group, order=20)
        # 滚动条数值
        self.num_slide = 0
        self.num_cover = Rectangle(
            x=37 + 8 * 65,
            y=self.height - 143,
            width=70,
            height=150,
            color=(255, 255, 255, 255),
            batch=self.main_batch,
            group=cover_group,
        )
        self.main_frame.add_calculate_func(
            self.num_cover,
            lambda rec, width, height, window: (37 + 8 * 65, height - 143),
        )
        # 从大到小
        num_group = Group(parent=self.num_group, order=10)
        for i in range(256):
            num_name = NumWidget(
                num=i, batch=self.main_batch, group=num_group, x=40, y=50
            )
            self.num_dict[i] = num_name
        self.num_hints = []
        # 每个部分的取值提示
        font = load_font("黑体", 15)
        font_height = font.ascent - font.descent
        num_hint_group = Group(parent=self.main_group, order=20)
        # hp: 3~6 len = 4
        # 要覆盖住 4 个数字
        self.num_hints.append(
            Rectangle(
                x=40 - 3,
                y=self.height - (173 + 30 * 6),
                width=62,
                height=(font_height + 4 + 5) * 4,
                # 浅蓝色背景
                color=(0, 0, 255, 100),
                batch=self.main_batch,
                group=num_hint_group,
            )
        )
        # 剩下 7 个, 每个都是中间
        for x in range(1, 8):
            self.num_hints.append(
                Rectangle(
                    x=40 - 3 + (65 * x),
                    y=self.height - (173 + 30),
                    width=62,
                    height=font_height + 4 + 5,
                    # 浅蓝色背景
                    color=(0, 0, 255, 100),
                    batch=self.main_batch,
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
        self.display_dict: Dict[NumStatus, List[NumWidget]] = {
            NumStatus.hp: [self.num_dict[i] for i in range(89, 99)],
            NumStatus.attack: [self.num_dict[i] for i in range(99, 102)],
            NumStatus.defense: [self.num_dict[i] for i in range(102, 105)],
            NumStatus.speed: [self.num_dict[i] for i in range(105, 108)],
            NumStatus.agility: [self.num_dict[i] for i in range(108, 111)],
            NumStatus.magic: [self.num_dict[i] for i in range(111, 114)],
            NumStatus.resistance: [self.num_dict[i] for i in range(114, 117)],
            NumStatus.wisdom: [self.num_dict[i] for i in range(117, 120)],
            NumStatus.wait: [self.num_dict[i] for i in range(120, 217)],
        }
        self.update_num_display()

    def update_num_display(self) -> None:
        # sort hp
        self.display_dict[NumStatus.hp].sort(key=lambda x: x.value)
        # sort wait
        self.display_dict[NumStatus.wait].sort(key=lambda x: x.value)

        for status, widgets in self.display_dict.items():
            num_count = 0
            if status == NumStatus.wait:
                continue
            for widget in widgets:
                widget.x = 40 + (65 * status.value)
                widget.y = self.height - (170 + 30 * num_count)
                num_count += 1

        # wait 的单独处理, 因为有滚动条
        num_count = 0
        for widget in self.display_dict[NumStatus.wait]:
            widget.x = 40 + (65 * NumStatus.wait.value)
            widget.y = self.height - (170 + 30 * num_count) + self.num_slide
            # 如果太高了, 就不显示了
            if widget.y > self.height - 200:
                # 给我不显示啊啊啊啊啊啊
                widget.display = False
            else:
                widget.display = True
            num_count += 1
        # 计算数据
        hp = sum(widget.value for widget in self.display_dict[NumStatus.hp][3:7]) + 154
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
        self.name_data = [
            attack,
            defense,
            speed,
            agility,
            magic,
            resistance,
            wisdom,
            hp,
        ]
        self.name_info_displays[
            "label"
        ].text = f"HP|{hp} 攻|{attack} 防|{defense} 速|{speed} 敏|{agility} 魔|{magic} 抗|{resistance} 智|{wisdom} 八围:{gather}"
        # 更新提示框
        # hp 提示框是固定的
        self.num_hints[0].y = self.height - (173 + 30 * 6)
        # 剩下的需要先判断那个是中间的
        for i in range(1, 8):
            data = sorted(
                enumerate(x.value for x in self.display_dict[NumStatus(i)]),
                key=lambda x: x[1],
            )
            middle_index = data[1][0]
            self.num_hints[i].y = self.height - (173 + 30 * middle_index)

    def init_name_dispaly(self) -> None:
        """
        初始化 名字显示 这块内容
        """
        name_group = Group(parent=self.main_group, order=30)
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
        if self.on_middle:
            # 正在滑动
            self.update_slide(self.drag_speed)
        self.main_batch.draw()

    def update_slide(self, dy: int) -> None:
        # 保证不会太下
        if (self.num_slide + dy) < 0:
            self.num_slide = 0
            self.update_num_display()
            return
        # 再保证不会太上
        num_len = len(self.display_dict[NumStatus.wait])
        if (self.num_slide + dy) > (30 * num_len - 100):
            self.num_slide = 30 * num_len - 100
            self.update_num_display()
            return
        self.num_slide += dy
        self.update_num_display()

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.update_slide(int(scroll_y) * -10)

    def on_mouse_press(self, x, y, button, modifiers):
        self.middle_base = (x, y)
        if not button & mouse.MIDDLE:  # 中键
            self.on_middle = False
        if button & mouse.LEFT:
            # 捏起
            if (x, y) in self.output_button:
                # 导出
                print("导出")
                for status, widgets in self.display_dict.items():
                    if status == NumStatus.wait:
                        continue
                    print(f"{status.name}: {', '.join(str(x.value) for x in widgets)}")
                name = self.name_info_displays["entry"].value
                print("名字: ", name)
                # +diy[50,51,90,130,150,70,89,210]
                print(f"{name}+diy{self.name_data}")
                return
            for idx, widget in self.num_dict.items():
                if widget.aabb(x, y):
                    self.drag_start = idx
                    # print(f"捏起 {idx}")
                    break

    def on_mouse_release(self, x, y, button, modifiers):
        self.on_middle = False
        if button & mouse.LEFT:
            find = []
            if self.drag_start:  # 有开始目标
                for idx, target_widget in self.num_dict.items():
                    if target_widget.aabb(x, y):
                        if idx == self.drag_start:
                            # 如果是自己, 就不做任何操作
                            continue
                        # 搜索对面那个, 修改双方显示内容
                        # 交换 value
                        find = [self.drag_start, idx]
                        break
            if find:
                print(f"交换 {find}")
                (
                    self.num_dict[find[0]].value,
                    self.num_dict[find[1]].value,
                ) = find[1], find[0]
                # 交换键
                self.num_dict[find[0]], self.num_dict[find[1]] = (
                    self.num_dict[find[1]],
                    self.num_dict[find[0]],
                )
            self.drag_start = None
            self.update_num_display()

    def on_mouse_leave(self, x, y):
        self.on_middle = False

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if dy != 0 and y != self.middle_base[1] and buttons & mouse.MIDDLE:
            if not self.on_middle:
                self.middle_base = (x, y)
                self.drag_speed = 0
                self.on_middle = True
                return
            drag_y = y - self.middle_base[1]
            # 取个对数, 保证不会太快
            self.drag_speed = int(drag_y)
        if self.drag_start:
            # 拖动
            self.num_dict[self.drag_start].x = x - 17
            self.num_dict[self.drag_start].y = y - 7

    def on_resize(self, width, height):
        super().on_resize(width, height)
        self.update_num_display()

    def start(self) -> None:
        pyglet.app.run(interval=1 / 30)


if __name__ == "__main__":
    window = MainWindow()
    glClearColor(1, 1, 1, 1)
    window.start()
