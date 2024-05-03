import os
from multiprocessing.shared_memory import SharedMemory
from muxserve.muxsched.workload_utils import Request
from typing import List, Union
import pickle
import copy

PREFIX = os.environ.get("FLEXSM_SHM_PREFIX", "")


def map_shm_name(shm_name):
    shm_name = PREFIX + shm_name.split("/")[-1]
    return shm_name


def create_shared_var(shm_name: str,
                      size: int = 0,
                      create: bool = True) -> SharedMemory:
    shm_name = map_shm_name(shm_name)
    # first close and unlink the shared memory if it exists and create is True
    if create:
        try:
            shm = SharedMemory(shm_name)
            shm.close()
            shm.unlink()
        except:
            pass

    shm = SharedMemory(shm_name, create=create, size=size)
    if create:
        shm.buf[:] = b"0" * size
    return shm


def read_shared_var(shm: SharedMemory) -> int:
    data = int(bytes(shm.buf[:]).decode('utf-8'))
    return data


def write_shared_var(shm: SharedMemory,
                     data: int,
                     close: bool = False) -> None:
    sign = "-" if data < 0 else "+"
    data = str(abs(data))
    data = sign + "0" * (shm.size - len(data) - 1) + data
    shm.buf[:] = data.encode('utf-8')
    if close:
        shm.close()


def dump_reqs_to_shared_var(shm_name: str, data: List[Request]) -> None:
    shm_name = map_shm_name(shm_name)

    # Serialize the list of Request objects
    serialized_data = pickle.dumps(copy.deepcopy(data))

    shm = SharedMemory(shm_name, create=True, size=len(serialized_data))

    # Write the serialized data to the shared memory
    shm.buf[:len(serialized_data)] = serialized_data
    shm.close()


def load_reqs_from_shared_var(shm_name: str) -> List[Union[Request, int]]:
    shm_name = map_shm_name(shm_name)
    try:
        shm = SharedMemory(shm_name)
    except:
        data = []
        return data

    # Read the serialized data from the shared memory
    serialized_data = bytes(shm.buf[:])

    # Deserialize the data back into a list of Request objects
    data = pickle.loads(serialized_data)

    shm.close()
    shm.unlink()

    return data


def dump_to_shared_var(shm_name: str, data: List[int]) -> None:
    shm_name = map_shm_name(shm_name)
    data = ",".join([str(x) for x in data])
    shm = SharedMemory(shm_name, create=True, size=len(data))
    shm.buf[:] = data.encode('utf-8')
    shm.close()


def load_from_shared_var(shm_name: str) -> List[int]:
    shm_name = map_shm_name(shm_name)
    try:
        shm = SharedMemory(shm_name)
    except:
        data = []
        return data
    data = bytes(shm.buf[:]).decode('utf-8')
    while "\x00" in data:
        data = bytes(shm.buf[:]).decode('utf-8')
    data = [int(x) for x in data.split(",")]
    shm.close()
    shm.unlink()
    return data


def write_str_to_shared_var(shm_name: str, data: str) -> None:
    shm_name = map_shm_name(shm_name)
    shm = SharedMemory(shm_name, create=True, size=len(data))
    shm.buf[:] = data.encode('utf-8')
    shm.close()


def read_str_from_shared_var(shm_name: str) -> str:
    shm_name = map_shm_name(shm_name)
    try:
        shm = SharedMemory(shm_name)
    except:
        data = ""
        return data
    data = bytes(shm.buf[:]).decode('utf-8')
    while "\x00" in data:
        data = bytes(shm.buf[:]).decode('utf-8')
    shm.close()
    return data


def write_list_to_shared_var(shm_name: str, data: List[int]) -> None:
    shm_name = map_shm_name(shm_name)
    data = ",".join([str(x) for x in data])
    shm = SharedMemory(shm_name, create=True, size=len(data))
    shm.buf[:] = data.encode('utf-8')
    shm.close()


def read_list_from_shared_var(shm_name: str) -> str:
    shm_name = map_shm_name(shm_name)
    try:
        shm = SharedMemory(shm_name)
    except:
        data = []
        return data
    data = bytes(shm.buf[:]).decode('utf-8')
    while "\x00" in data:
        data = bytes(shm.buf[:]).decode('utf-8')
    data = [int(x) for x in data.split(",")]
    shm.close()
    return data


def close_shared_var(shm_name: str) -> None:
    shm_name = map_shm_name(shm_name)
    try:
        shm = SharedMemory(shm_name)
        shm.close()
        shm.unlink()
    except:
        pass
