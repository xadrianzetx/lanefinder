import cv2
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.image import Image
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.actionbar import ActionBar, ActionView, ActionOverflow
from kivy.uix.actionbar import ActionPrevious, ActionButton, ActionGroup
from kivy.graphics.texture import Texture
from .utils import rgba_to_kivy


class VideoFeed(Image):

    def __init__(self, capture, fps, **kwargs):
        Image.__init__(self, **kwargs)
        self._capture = capture
        self.allow_stretch = True
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        # read next frame from camera on clock tick
        ret, frame = self._capture.read()
        height, width, _ = frame.shape

        if ret:
            # TODO net inference
            buff = cv2.flip(frame, 0)
            buff = buff.tostring()

            # create buffer from frame
            img_texture = Texture.create(size=(width, height), icolorfmt='bgr')
            img_texture.blit_buffer(buff, colorfmt='bgr', bufferfmt='ubyte')

            # set texture
            self.texture = img_texture


class MainMenuActionPrevious(ActionPrevious):

    def __init__(self, **kwargs):
        ActionPrevious.__init__(self, **kwargs)

    @property
    def app_icon(self):
        # 32x32 pixel icon location
        return 'lanefinder/assets/icons/icon.png'

    @property
    def with_previous(self):
        return False


class ButtonMode(ActionButton):

    def __init__(self, **kwargs):
        ActionButton.__init__(self, **kwargs)
        self._text_options = ['CPU', 'TPU']
        self.background_down = ''
        self.text = self._text_options[0]

    @property
    def on_press(self):
        # change background color and return noop as callable
        self.background_color = rgba_to_kivy([203, 0, 0, 1])
        return lambda *args: None

    @property
    def on_release(self):
        return self.update

    def update(self):
        self._text_options = self._text_options[::-1]
        self.text = self._text_options[0]
        # TODO swich TPU/CPU mode
        return lambda *args: None


class ButtonExit(ActionButton):

    def __init__(self, **kwargs):
        ActionButton.__init__(self, **kwargs)
        self.background_down = ''
        self.text = 'Exit'

    @property
    def on_press(self):
        # change background color and return noop as callable
        self.background_color = rgba_to_kivy([203, 0, 0, 1])
        return lambda *args: None

    @property
    def on_release(self):
        # terminate app - callable
        return App.get_running_app().stop


class MainMenuActionGroup(ActionGroup):

    def __init__(self, **kwargs):
        ActionGroup.__init__(self, **kwargs)
        self.add_widget(ButtonMode())
        self.add_widget(ButtonExit())


class MainMenuActionView(ActionView):

    def __init__(self, **kwargs):
        ActionView.__init__(self, **kwargs)
        self.action_previous = MainMenuActionPrevious()
        self.action_group = MainMenuActionGroup()
        self.add_widget(self.action_previous)
        self.add_widget(self.action_group)


class MainMenuBar(ActionBar):

    def __init__(self, **kwargs):
        ActionBar.__init__(self, **kwargs)
        self.pos_hint = 1
        self.background_image = ''
        self.background_color = rgba_to_kivy([207, 55, 33, 1])
        self.view = MainMenuActionView()
        self.add_widget(self.view)


class ScreenLayout(GridLayout):

    def __init__(self, **kwargs):
        GridLayout.__init__(self, **kwargs)
        self.rows = 2
        self.capture = cv2.VideoCapture(0)
        self.feed = VideoFeed(capture=self.capture, fps=30)
        self.menu = MainMenuBar()
        self.add_widget(self.feed)
        self.add_widget(self.menu)
