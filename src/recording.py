import sys
import json

class Recorder:
    def __init__(self, eid : str, print_every = 1, dump_every = None, savedir = "."):
        self.print_every = print_every
        self.dump_every = dump_every
        self.savedir = savedir
        self.raw_filename = savedir + "/" + eid + ".yml"
        self.indent = "  "

    def dump_global_header(self, header : dict):
        with open(self.raw_filename, "w") as fp:
            for k, v in header.items():
                fp.write(f"\"{k}\": {v}\n")
            fp.write(f"\"training_data\":\n")

    def dump_state(self, state : dict):
        with open(self.raw_filename, "a") as fp:
            fp.write(self.indent * 2 + "- ")
            json.dump(state, fp)
            fp.write("\n")

    def should_print(self, i : int) -> bool:
        return (i == 0) or ( (i+1) % self.print_every == 0)

    def should_dump(self, i : int) -> bool:
        return (i == 0) or ( (i+1) % self.dump_every == 0)

    def teardown(self):
        sys.stdout.write(f"\r\033[0K")
        sys.stderr.write(f"\r\033[0K")
