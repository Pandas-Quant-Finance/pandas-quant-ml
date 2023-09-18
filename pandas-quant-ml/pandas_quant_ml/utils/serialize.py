import io

import dill


def serialize(obj, filename):
    if isinstance(filename, io.BufferedIOBase):
        dill.dump(obj, filename)
    else:
        with open(str(filename), 'wb') as file:
            dill.dump(obj, file)


def deserialize(filename):
    if isinstance(filename, io.BufferedIOBase):
        return dill.load(filename)
    else:
        with open(str(filename), 'rb') as file:
            return dill.load(file)
