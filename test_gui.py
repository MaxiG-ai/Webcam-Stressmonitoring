import re
import threading
from frontend import gui, IFrontend


SET_HEARTRATE_COMMAND = re.compile(r"set_heartrate (\d+)")


def backend(frontend: IFrontend):
    while True:
        user_input = input("> ")
        if user_input == "quit":
            break
        elif user_input == "get_image":
            image = frontend.get_image()
            print(image)
        elif SET_HEARTRATE_COMMAND.match(user_input):
            heartrate = int(SET_HEARTRATE_COMMAND.match(user_input).group(1))
            frontend.set_heartrate(heartrate)


if __name__ == "__main__":
    gui = gui.Gui()

    backend_thread = threading.Thread(target=backend, args=[gui])
    backend_thread.daemon = True
    backend_thread.start()

    gui.start()