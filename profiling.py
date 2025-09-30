import os
import time
from contextlib import contextmanager
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler


class ThroughputMeter:
    def __init__(self, drop_first_steps: int = 0):
        self.drop_first_steps = drop_first_steps
        self._steps = 0
        self._samples = 0
        self._t0 = None

    def update(self, batch_size: int):
        self._steps += 1
        if self._steps <= self.drop_first_steps:
            return
        if self._t0 is None:
            self._t0 = time.time()
        self._samples += batch_size

    def rate(self) -> float:
        if self._t0 is None:
            return 0.0
        dt = time.time() - self._t0
        return self._samples / dt if dt > 0 else 0.0



def make_profiler(logdir: str,
                  wait: int = 5,
                  warmup: int = 10,
                  active: int = 50,
                  with_stack: bool = True,
                  record_shapes: bool = True,
                  profile_memory: bool = True):
    """
    Возвращает кортеж (prof_context, total_steps), где:
      - prof_context - контекстный менеджер профайлера
      - total_steps - общее число шагов, которое надо выполнить (wait+warmup+active)
    """
    def _activities():
        acts = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            acts.append(ProfilerActivity.CUDA)
        return acts
    
    os.makedirs(logdir, exist_ok=True)
    sched = schedule(wait=wait, warmup=warmup, active=active, repeat=1)
    activities = _activities()
    prof = profile(
        activities=activities,
        schedule=sched,
        on_trace_ready=tensorboard_trace_handler(logdir),
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    )
    total = (wait + warmup + active)
    return prof, total


def print_top_tables(prof, row_limit: int = 40):
    print("\n==== TOP CUDA ops (self_cuda_time_total) ====")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=row_limit))
    print("\n==== TOP CPU ops (self_cpu_time_total) ====")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=row_limit))

    try:
        print("\n==== TOP CUDA ops (group_by_input_shape) ====")
        print(prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=row_limit))
    except Exception:
        pass


def export_trace(prof, path: str = "./trace.json"):
    try:
        prof.export_chrome_trace(path)
        print(f"trace сохранён в: {path}")
    except Exception as ex:
        print(f"Не удалось сохранить trace: {ex}")


@contextmanager
def marked(name: str):
    """
    Чтобы в трейсах видеть по блокам
    """
    with record_function(name):
        yield
