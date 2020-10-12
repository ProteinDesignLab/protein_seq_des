from tensorboardX import SummaryWriter
import numpy as np
import os
import datetime
import subprocess


class Logger(object):
    def __init__(self, log_dir="./logs", dummy=False, prefix="", suffix="", full_log_dir=None, rank=0):
        self.suffix = suffix
        self.prefix = prefix
        self.dummy = dummy
        if self.dummy:
            return

        self.iteration = 1

        if log_dir == "":
            log_dir = "./logs"
        if full_log_dir is None or log_dir == "":
            now = datetime.datetime.now()
            self.ts = now.strftime("%Y-%m-%d-%H-%M-%S")
            log_path = os.path.join(log_dir, self.prefix + self.ts + self.suffix)
        else:
            log_path = full_log_dir

        self.log_path = log_path
        if not os.path.isdir(log_path):
            self.writer = SummaryWriter(log_dir=log_path)
        self.kvs = {}

        print(("Logging to", log_path))

    def log_args(self, args):
        with open("%s/args.txt" % self.log_path, "w") as f:
            for arg in vars(args):
                f.write("%s\t%s\n" % (arg, getattr(args, arg)))

    def advance_iteration(self):
        self.iteration += 1

    def reset_iteration(self):
        self.iteration = 0

    def log_scalar(self, name, value):
        if self.dummy:
            return

        if isinstance(value, list):
            assert len(value) == 1, (name, len(value), value)
            return self.log_scalar(name, value[0])
        try:
            self.writer.add_scalar(name, value, self.iteration)
        except Exception as e:
            print(("Failed on", name, value, type(value)))
            raise

    def log_kvs(self, **kwargs):
        if self.dummy:
            return

        for k, v in kwargs.items():
            assert isinstance(k, str)
            self.kvs[k] = v

        kv_strings = ["%s=%s" % (k, v) for k, v in sorted(self.kvs.items())]
        val = "<br>".join(kv_strings)
        self.writer.add_text("properties", val, global_step=self.iteration)
