import time
import torch
import traceback
from ipc_tensor.zmq_tool import ZMQServer, timestamp

PORT = 10050

test_tensor = torch.Tensor([[1, 2, 3], [4, 5, 6]]).cuda()


def main():
    print(test_tensor)
    test_tensor_1 = test_tensor[1:2, 0:2]
    print(test_tensor.size(), test_tensor.stride(),
          test_tensor.storage_offset())
    print(test_tensor_1.size(), test_tensor_1.stride(),
          test_tensor_1.storage_offset())

    storage = test_tensor.storage()
    storage_1 = test_tensor_1.storage()
    storage = test_tensor.storage()
    '''
    (storage_device, storage_handle, storage_size_bytes, storage_offset_bytes,
    [require_grad], ref_counter_handle, ref_counter_offset, event_handle, event_sync_required)
    '''
    t = storage._share_cuda_()
    tcp_server = ZMQServer('localhost', PORT)

    while True:
        timestamp('socket opened, waiting', 'server')

        data = tcp_server.recv_string()
        timestamp(f'received from client: {data}', 'server')

        params_dict = {
            "tensor_size": test_tensor.size(),
            "tensor_stride": test_tensor.stride(),
            "tensor_offset": test_tensor.storage_offset(),
            "storage_cls": type(storage),
            "dtype": test_tensor.dtype,
            "storage_device": t[0],
            "storage_handle": t[1],
            "storage_size_bytes": t[2],
            "storage_offset_bytes": t[3],
            "requires_grad": test_tensor.requires_grad,
            "ref_counter_handle": t[4],
            "ref_counter_offset": t[5],
            "event_handle": t[6],
            "event_sync_required": t[7]
        }
        time.sleep(1)
        timestamp('sending metadata of tensor storage', 'server')
        tcp_server.send_pyobj(params_dict)

        time.sleep(1)
        data = tcp_server.recv_string()
        timestamp(f'received signal from client, printing: {data}', 'server')
        print(test_tensor)

        time.sleep(1)
        test_tensor[1, :] = 0
        timestamp('server modified tensor, sending signal to client', 'server')
        tcp_server.send_string('MODF')

        time.sleep(1)
        timestamp('printing final tensor, and waiting for client signal',
                  'server')
        print(test_tensor)

        data = tcp_server.recv_string()
        timestamp(f'received signal from client, bye: {data}', 'server')
        tcp_server.send_string('bye')
        if data == 'break':
            break


if __name__ == '__main__':
    """
    Usage: PYTHONPATH=ipc_tensor:$PYTHONPATH python ipc_tensor/zmq_server.py
    """
    try:
        main()
    except Exception:
        traceback.print_exc()
    finally:
        print('del')
        print(test_tensor)
        del test_tensor
