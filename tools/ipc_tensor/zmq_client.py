import sys
import time
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor
from ipc_tensor.zmq_tool import ZMQClient, timestamp

PORT = 10050


def main():
    timestamp('connecting server', 'client')
    tcp_client = ZMQClient('localhost', PORT)

    tcp_client.send_string('hello')

    timestamp('connected, waiting data', 'client')
    data = tcp_client.recv_pyobj()
    print(data)

    timestamp('data received, rebuilding and printing', 'client')
    test_tensor = rebuild_cuda_tensor(torch.Tensor, **data)
    print(test_tensor)

    time.sleep(1)
    timestamp('modifying tensor and notify server', 'client')
    test_tensor[:, 1] = 8
    print(test_tensor)
    tcp_client.send_string('MODF')

    time.sleep(1)
    timestamp('waiting for server sending signal', 'client')
    _ = tcp_client.recv_string()

    timestamp('signal received, print', 'client')
    print(test_tensor)

    time.sleep(1)
    timestamp('prepare to quit', 'client')
    del test_tensor
    tcp_client.send_string(sys.argv[1])


if __name__ == '__main__':
    """
    Usage: PYTHONPATH=ipc_tensor:$PYTHONPATH python ipc_tensor/zmq_client.py break
    """
    assert len(sys.argv) > 1
    main()
