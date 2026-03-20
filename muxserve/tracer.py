import json

from muxserve.logger import init_logger

logger = init_logger(__name__)


def pack_to_proc_name(model_name, mps_percentage):
    model_name = model_name.split("/")[-1]
    return f"{model_name} (MPS {mps_percentage})"


class FlexTracer:

    def __init__(self, process_names):
        self.trace_events = []

        self.proc_to_pid = {}
        for pid, proc_name in enumerate(process_names):
            self.proc_to_pid[proc_name] = pid
            meta_event = {
                'name': 'thread_name',
                'ph': 'M',
                'pid': pid,
                'tid': 0,
                'args': {
                    'name': f'{proc_name}'
                }
            }
            self.trace_events.append(meta_event)
            meta_event = {
                'name': 'thread_sort_index',
                'ph': 'M',
                'pid': pid,
                'tid': 0,
                'args': {
                    'sort_index': pid,
                }
            }
            self.trace_events.append(meta_event)

        self.event_holder = {}

    def add_trace_event(self, name, cat, proc_name, times):
        for ph, ts in zip(['B', 'E'], times):
            event = {
                'name': name,
                'cat': cat,
                'ph': ph,
                'pid': self.proc_to_pid[proc_name],
                'tid': 0,
                'ts': ts * 1e6,
            }
            self.trace_events.append(event)

    def add_event(self, name, cat, proc_name, ts, start=True):
        if start:
            self.event_holder[proc_name] = ts
            return

        st_tick = self.event_holder.pop(proc_name)
        times = [st_tick, ts]
        self.add_trace_event(name, cat, proc_name, times)

    def export(self, trace_file):
        logger.info(f"Export execution trace to {trace_file}")
        trace_data = {
            'traceEvents': self.trace_events,
            'displayTimeUnit': 'ms'
        }
        with open(f'{trace_file}', 'w') as f:
            json.dump(trace_data, f)
