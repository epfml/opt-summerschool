# by Thijs Vogels

import json
import os
import pickle
import random
from collections import defaultdict
from datetime import datetime, timedelta
from glob import glob
from string import Template
from threading import Event, Thread
from time import sleep, time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from IPython.core.display import Javascript, clear_output, display
from IPython.core.magic import register_cell_magic
from IPython.display import set_matplotlib_formats
from PIL import Image
from pymongo import MongoClient
from redis import StrictRedis
from torch.multiprocessing import Process

WORKER_ID = os.getenv("WORKER_ID")


mpl.rcParams["savefig.dpi"] = 80
mpl.rcParams["figure.dpi"] = 80


print(f"Initializing the tutorial environment for user {WORKER_ID} ...")


REDIS_IS_AVAILABLE = os.getenv("REDIS_MASTER_SERVICE_HOST") is not None

if REDIS_IS_AVAILABLE:
    redis = StrictRedis(
        host=os.getenv("REDIS_MASTER_SERVICE_HOST"),
        port=int(os.getenv("REDIS_MASTER_SERVICE_PORT")),
        db=0,
        password=os.getenv("REDIS_PASSWORD"),
    )


def answer(question_id, answer):
    """Store a worker's answer to a question"""
    redis.hset(f"question:{question_id}", WORKER_ID, answer)


def get_all_answers(question_id):
    """Get a dictionary of worker_id: answer for everyone who answered a question"""
    return redis.hgetall(f"question:{question_id}")


def get_worker_answer(question_id, worker_id):
    """Get the answer to a question from a specific user"""
    return redis.hget(f"question:{question_id}", worker_id)


def set_global(variable_name, value):
    """Set a value that is shared among all workers"""
    if isinstance(value, bool):
        value = 1 if value else 0
    redis.set(f"global:{variable_name}", value)


def set_worker_value(variable_id, value, worker_id=WORKER_ID):
    redis.set(f"value:{worker_id}:{variable_id}", pickle.dumps(value))


def get_worker_value(variable_id, worker_id=WORKER_ID):
    result = redis.get(f"value:{worker_id}:{variable_id}")
    if result is not None:
        return pickle.loads(result)
    else:
        return None


def get_global(variable_name, type=str):
    """Get a value that is shared among all workers"""
    out = redis.get(f"global:{variable_name}")
    if not out:
        return None
    else:
        if type == str:
            return out.decode("utf-8")
        else:
            return type(out)


def send(worker_id, message_id, value):
    """Send a personal message to a worker, to be received with `receive`"""
    if isinstance(value, bool):
        value = 1 if value else 0
    redis.rpush(f"message:{worker_id}:{message_id}", value)


def receive(message_id, blocking=False, type=str):
    """Receive a personal message that was sent with `send`"""
    message_key = f"message:{WORKER_ID}:{message_id}"
    if blocking:
        key, message = redis.blpop(message_key)
    else:
        message = redis.lpop(message_key)
        if not message:
            return None

    if type == str:
        return message.decode("utf-8")
    else:
        return type(message)


def clear_all_messages():
    """This is very dangerous. Clear redis completely"""
    for key in redis.keys():
        redis.delete(key)


def experiment_enable(experiment_name):
    """To be run by the organizer to allow others to join an experiment"""
    experiment_reset(experiment_name)

    set_global(f"experiment:{experiment_name}:enabled", True)

    # Register master node
    redis.zadd(f"experiment:{experiment_name}:participants", {WORKER_ID: 0})


def experiment_join(experiment_name, your_code, skip_tests=False):
    """Join an experiment, passing a function for your experiment code"""
    if not skip_tests:
        experiment_is_enabled = get_global(f"experiment:{experiment_name}:enabled", type=bool)
        if not experiment_is_enabled:
            raise RuntimeError("It is too early to run this experiment. Please wait for a bit.")

        experiment_is_running = get_global(f"experiment:{experiment_name}:running", type=bool)
        if experiment_is_running:
            raise RuntimeError(
                "This experiment is already running. You are too late to join. Sorry."
            )

    # Register ourself
    redis.zadd(f"experiment:{experiment_name}:participants", {WORKER_ID: 1})

    def fn():
        print("Waiting for more workers to join. Thanks for your patience ...")

        # Wait until we receive a rank from the master node
        rank = receive(f"experiment:{experiment_name}:rank", blocking=True, type=int)
        world_size = receive(f"experiment:{experiment_name}:world_size", blocking=True, type=int)

        pytorch_distributed_init(experiment_name, rank, world_size)
        print(f"Connected to {world_size} workers. You are worker number {rank}. Starting ...")

        start_time = time()
        your_code()
        duration = time() - start_time

        print(f"Execution finished in {duration:.1f}s.")

    p = Process(target=fn)
    p.start()
    p.join()


def pytorch_distributed_init(experiment_name, rank, world_size):
    # First undo previous initializations
    try:
        torch.distributed.destroy_process_group()
    except RuntimeError:
        pass  # it this didn't work, it's fine. it was non-existent then

    torch.distributed.init_process_group(
        "gloo",
        f"file:///shared/experiment_{experiment_name}",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(0, 60),
    )


def experiment_launch(experiment_name, your_code):
    """To be run by the organizers. This is the start sign of an experiment"""
    experiment_is_enabled = get_global(f"experiment:{experiment_name}:enabled", type=bool)
    if not experiment_is_enabled:
        raise RuntimeError("It is too early to run this experiment. Please wait for a bit.")

    experiment_is_running = get_global(f"experiment:{experiment_name}:running", type=bool)
    if experiment_is_running:
        raise RuntimeError("This experiment is already running. You are too late to join. Sorry.")

    participants_key = f"experiment:{experiment_name}:participants"
    participants = [b.decode("utf-8") for b in redis.zrange(participants_key, 0, -1)]
    master_node = participants[0]
    if not WORKER_ID == master_node:
        print("Only the master is allowed to start an experiment")

    # Set the state to running
    set_global(f"experiment:{experiment_name}:running", True)

    # Send everyone their rank and world size, the sign to start
    world_size = len(participants)
    for rank, participant in enumerate(participants):
        send(participant, f"experiment:{experiment_name}:rank", rank)
        send(participant, f"experiment:{experiment_name}:world_size", world_size)

    experiment_join(experiment_name, your_code, skip_tests=True)


def experiment_reset(experiment_name):
    """Restart from scratch with the experiment"""
    # Delete references in the database
    redis.delete(f"experiment:{experiment_name}:participants")
    redis.delete(f"global:experiment:{experiment_name}:enabled")
    redis.delete(f"global:experiment:{experiment_name}:running")

    # Delete file on the shared filesystem
    join_file = f"/shared/experiment_{experiment_name}"
    if os.path.isfile(join_file):
        os.unlink(join_file)


class IntervalThread(Thread):
    def __init__(self, fn, stop_condition=None, interval=1.0):
        self.stop_event = Event()

        def thread_fn():
            fn()
            while not self.stop_event.wait(interval):
                if stop_condition is not None and stop_condition():
                    return
                fn()

        self.thread = Thread(target=thread_fn)
        self.thread.start()

    def stop(self):
        self.stop_event.set()


class ReactCell:
    def __init__(self):
        self.cell = "cell{}".format(random.randint(1_000_000_000, 9_999_999_999))
        execute_js(
            f"window.{self.cell}=document.createElement('div'); element[0].appendChild(window.{self.cell});"
        )  # Make a global variable to access this cell

    def render(self, jsx_string):
        jsx(0, f"""ReactDOM.render({jsx_string}, window.{self.cell});""")


class ActiveWorkerMonitor:
    def __init__(self, interval=10):
        self.cell = ReactCell()

        def update_count():
            count = len(active_workers())
            self.cell.render(f"""<p>Active users: <strong>{count}</strong></p>""")

        self.thread = IntervalThread(fn=update_count, interval=interval)

    def stop(self):
        self.thread.stop()
        count = len(active_workers())
        self.cell.render(f"""<p>Active users: <strong>{count}</strong> (non-updating)</p>""")


class Questions:
    def __init__(self, *questions):
        self.cell = ReactCell()
        html = "<div>"
        for question_id, label, options in questions:
            question_html = "<div style={{marginBottom: '1em'}}>"
            question_html += label + "<br />"
            for option in options:
                question_html += (
                    """<label>"""
                    + option
                    + """ <input type="radio" name="x"""
                    + question_id
                    + """x" onClick={(event) => python("tutorial.answer(\'"""
                    + question_id
                    + """\', \'"""
                    + option
                    + """\')")} /></label>&nbsp;&nbsp;&nbsp;"""
                )
            question_html += "</div>"
            html += question_html
        html += "</div>"
        self.cell.render(html)


class AnswerMonitor:
    def __init__(self, *questions):
        self.questions = questions
        self.cell = ReactCell()
        self.thread = IntervalThread(fn=self.render, interval=5)

    def render(self):
        html = "<div>"
        for question_id, label, options in self.questions:
            active_users = set(active_workers())
            n_active = len(active_users)
            answers = get_all_answers(question_id)
            tally = {option: 0 for option in options}
            total_count = 0
            for user, answer in answers.items():
                answer = answer.decode("utf-8")
                if answer in tally and user.decode("utf-8") in active_users:
                    tally[answer] += 1
                    total_count += 1
            tally["no_vote"] = n_active - total_count

            question_html = "<div style={{display: 'flex', marginBottom: '.4em'}}>"
            question_html += "<div style={{width: '18em'}}>" + label + "</div>"
            question_html += "<div style={{flexGrow: 1, backgroundColor: '#eee', display: 'flex'}}>"
            for option in options:
                bg_color = {"yes": "#bada55", "no": "#ff5534", "3": "#55bada", "4": "#daba55"}[
                    option
                ]
                color = "black"
                if tally[option] > 0:
                    question_html += (
                        "<div style={{width: '0px', color: '"
                        + color
                        + "', backgroundColor: '"
                        + bg_color
                        + "', flexShrink: 1, flexBase: 1, overflow: 'hidden', flexGrow: "
                        + str(tally[option])
                        + "}}>&nbsp;"
                        + option
                        + "</div>"
                    )
            if tally["no_vote"] > 0:
                question_html += (
                    "<div style={{width: '0px', flexShrink: 1, flexBase: 1, overflow: 'hidden', flexGrow: "
                    + str(tally["no_vote"])
                    + "}}></div>"
                )
            question_html += "</div>"
            question_html += "</div>"
            html += question_html
        html += "</div>"
        self.cell.render(html)

    def stop(self):
        self.thread.stop()


def experiment_monitor_participants(experiment_name):
    cell = ReactCell()

    def update_cell():
        n_participants = redis.zcount(f"experiment:{experiment_name}:participants", 0, 1)
        cell.render(f"""<p>Registered participants: <strong>{n_participants}</strong></p>""")

    return IntervalThread(
        fn=update_cell,
        stop_condition=lambda: get_global(f"experiment:{experiment_name}:running", type=bool),
    )


def execute_js(code):
    """Execute JavaScript code in a Notebook environment"""
    display(Javascript(code))
    # clear_output(wait=True)


def start_js_message_thread():
    """This listens for JavaScript code coming in from the organizers, and executes whatever it receives"""

    def listener():
        while True:
            key, message = redis.blpop("js_inbox:" + WORKER_ID)
            message = message.decode("utf-8")
            try:
                execute_js(message)
            except Exception as e:
                pass

    thread = Thread(target=listener)
    thread.start()
    return thread


def start_py_message_thread():
    """This listens for Python code coming in from the organizers, and executes whatever it receives"""

    def listener():
        while True:
            key, message = redis.blpop("py_inbox:" + WORKER_ID)
            message = message.decode("utf-8")
            try:
                exec(message)
            except Exception as e:
                pass

    thread = Thread(target=listener)
    thread.start()
    return thread


def start_heartbeat_thread(interval=10):
    """This regularly reports that a user is still alive"""

    def listener():
        while True:
            redis.hset("heartbeat", WORKER_ID, datetime.now().timestamp())
            sleep(10)

    thread = Thread(target=listener)
    thread.start()
    return thread


def start_threads():
    start_js_message_thread()
    start_py_message_thread()
    start_heartbeat_thread()


if REDIS_IS_AVAILABLE:
    start_threads()


def active_workers(heartbeat_treshold=20):
    """Get a list of worker IDs or running notebooks"""
    now = datetime.now().timestamp()
    active_workers = []
    for key, timestamp in redis.hgetall("heartbeat").items():
        if now - float(timestamp) < heartbeat_treshold:
            active_workers.append(key.decode("utf-8"))
    return active_workers


@register_cell_magic
def broadcast_py(line, cell, exclude_self=False):
    for worker in active_workers():
        if exclude_self and worker == WORKER_ID:
            continue
        print(f"Sending code to {worker}")
        redis.rpush(f"py_inbox:{worker}", cell)


@register_cell_magic
def broadcast_js(line, cell, exclude_self=False):
    for worker in active_workers():
        if exclude_self and worker == WORKER_ID:
            continue
        print(f"Sending code to {worker}")
        redis.rpush(f"js_inbox:{worker}", cell)


# This is to be able to render React code
display(
    Javascript(
        """
// Load our libraries from a CDN instead of wherever this notebook is hosted.
require.config({
    paths: {
        babel: '/files/lib/babel.min',
        react: '/files/lib/react.min',
        'react-dom': '/files/lib/react-dom.min'
    }
})

// Hook to call into Python.
// Credit to disarticulate for documenting the usage of iopub: 
//    https://gist.github.com/disarticulate/d06069ff3e71cf828e5329beab8cb084
window.python = code => new Promise((resolve, reject) => {
    IPython.notebook.kernel.execute(
        code,
        {iopub: {output: data => resolve(data.content.text)}},
    )   
})
"""
    )
)


class LivePlot:
    def __init__(self, legend, xmin=None, xmax=None, ymin=None, ymax=None):
        self.fig = plt.figure(figsize=(8, 5))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.points_x = []
        self.points_y = defaultdict(list)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.legend = legend
        self.colors = ["r", "g", "b"]

    def add_point(self, x, *ys):
        self.ax.cla()  # clear axis
        self.points_x.append(x)
        for i, y in enumerate(ys):
            self.points_y[i].append(y)

        self.ax.clear()
        handles = []
        for line_index, color, label in zip(self.points_y, self.colors, self.legend):
            x = np.array(self.points_x)
            y = np.array(self.points_y[line_index])

            nonzero = y > 0.0
            h = self.ax.semilogy(x[nonzero], y[nonzero], color, label=label)
            handles.append(h)
        if self.xmin is not None:
            self.ax.set_xlim([self.xmin, self.xmax])
        if self.ymin is not None:
            self.ax.set_ylim([self.ymin, self.ymax])
        self.ax.legend()

        display(self.fig)
        clear_output(wait=True)


pytorch_to_pil = torchvision.transforms.ToPILImage()  # reconvert into PIL image


def show_image(tensor, title=None, ax=None):
    image = tensor.squeeze(0).cpu()  # shape (3, h, w)
    image = pytorch_to_pil(image)
    if ax is not None:
        ax.imshow(image)
        ax.tick_params(
            # axis='x',          # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False,
        )  # labels along the bottom edge are off
        if title is not None:
            ax.set_xlabel(title)
    else:
        plt.imshow(image)
        plt.tick_params(
            # axis='x',          # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False,
        )  # labels along the bottom edge are off
        if title is not None:
            plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated


loader = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(
            256 if torch.cuda.is_available() else 128
        ),  # scale imported image
        torchvision.transforms.ToTensor(),  # transform it into a torch tensor
    ]
)


def load_image(image_name, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    image = Image.open(image_name)  # shape (3, height, width)
    image = loader(image).unsqueeze(0)[:, :3, :, :]  # shape (1, 3, height, width)
    return image.to(device, torch.float)


def load_styletransfer_images(directory="styletransfer"):
    image_paths = sorted(glob(os.path.join(directory, "*.jpg")))
    print(f"{len(image_paths)} images found. Loading ...")
    images = [load_image(i) for i in image_paths]
    print("Done loading.")
    return images


@register_cell_magic
def jsx(line, cell):
    display(
        Javascript(
            (
                Template(
                    """
                        require(['babel', 'react', 'react-dom'], (Babel, React, ReactDOM) => {
                            eval(Babel.transform($quoted_script, {presets: ['react']}).code)
                        })
                    """
                ).substitute(quoted_script=json.dumps(cell))
            )
        )
    )


def questionnaire():
    Questions(
        ("used_python", "Have you used Python?", ["yes", "no"]),
        ("used_pytorch", "Have you used PyTorch?", ["yes", "no"]),
        ("used_numpy", "Have you used NumPy?", ["yes", "no"]),
        ("used_tensorflow", "Have you used TensorFlow?", ["yes", "no"]),
        ("trained_neural_net", "Have you ever trained a neural net?", ["yes", "no"]),
        ("cnn", "What is the dimensionality of parameter tensors in a CNN layers?", ["3", "4"]),
        ("discuss_backprop", "Would you like us to discuss the backprop algorithm?", ["yes", "no"]),
    )


def lecture_questionnaire():
    Questions(
        ("lecture_like", "Did you like the lecture?", ["no", "meh", "yes", "a lot"]),
        (
            "lecture_difficulty",
            "Was the difficulty of the lecture?",
            ["too easy", "good", "too difficult"],
        ),
        ("lecture_speed", "How was the speed of the lecture", ["too slow", "good", "too fast"]),
        ("learn_new", "Did you learn something new?", ["yes", "no"]),
    )


def tutorial_questionnaire():
    Questions(
        ("tutorial_like", "Did you like the practical?", ["no", "meh", "yes", "a lot"]),
        (
            "tutorial_difficulty",
            "Was the difficulty of the practical?",
            ["too easy", "good", "too difficult"],
        ),
        ("tutorial_speed", "How was the speed of the practical", ["too slow", "good", "too fast"]),
    )


def questionnaire_answers():
    return AnswerMonitor(
        ("used_python", "Have you used Python?", ["yes", "no"]),
        ("used_tensorflow", "Have you used TensorFlow?", ["yes", "no"]),
        ("used_pytorch", "Have you used PyTorch?", ["yes", "no"]),
        ("used_numpy", "Have you used NumPy?", ["yes", "no"]),
        ("trained_neural_net", "Have you ever trained a neural net?", ["yes", "no"]),
        ("cnn", "CNN dimensionality?", ["3", "4"]),
        ("discuss_backprop", "Discuss backprop?", ["yes", "no"]),
    )


# questionnaire()


# _________ Helper functions _________


class TwoClassCIFAR(torchvision.datasets.CIFAR10):
    def __init__(self, train=True, transform=None):
        super(TwoClassCIFAR, self).__init__(
            root="data", train=train, download=True, transform=transform
        )
        indexes = np.arange(len(self.targets))[
            (np.array(self.targets) == 0) + (np.array(self.targets) == 1)
        ]
        self.data = self.data[indexes]
        self.targets = list(np.array(self.targets)[indexes])


# _________ Resnet (SmallNet) Architecture _______________


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_batchnorm=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if not use_batchnorm:
            self.bn1 = self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes) if use_batchnorm else nn.Sequential(),
            )

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, use_batchnorm=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        if not use_batchnorm:
            self.bn1 = self.bn2 = self.bn3 = nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes) if use_batchnorm else nn.Sequential(),
            )

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = torch.nn.functional.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out


class SmallResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_batchnorm=True):
        super(SmallResNet, self).__init__()
        reduction = 2
        self.in_planes = 64 // reduction
        self.use_batchnorm = use_batchnorm
        self.conv1 = nn.Conv2d(3, 64 // reduction, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64 // reduction) if use_batchnorm else nn.Sequential()
        self.layer1 = self._make_layer(block, 64 // reduction, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128 // reduction, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256 // reduction, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512 // reduction, num_blocks[3], stride=2)
        self.linear = nn.Linear((512 // reduction) * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.use_batchnorm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def SmallNet(num_classes=10, use_batchnorm=False):
    return SmallResNet(
        BasicBlock, [1, 1, 1, 1], num_classes=num_classes, use_batchnorm=use_batchnorm
    )


def get_vgg19_trained_weights():
    model = torchvision.models.vgg19(pretrained=True).features
    return model.state_dict()


set_matplotlib_formats("retina")
