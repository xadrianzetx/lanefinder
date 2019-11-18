import cv2
from kivy.app import App
from lanefinder.frontend import ScreenLayout


class Lanefinder(App):

    def build(self):
        # load main app layout
        self.layout = ScreenLayout()
        return self.layout

    def on_start(self):
        # check if periferals are connected
        pass

    def on_stop(self):
        # release camera
        self.layout.capture.release()


if __name__ == "__main__":
    Lanefinder().run()
