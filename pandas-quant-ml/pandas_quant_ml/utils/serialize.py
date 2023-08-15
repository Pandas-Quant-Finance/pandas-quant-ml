import dill


def serialize(obj, filename):
    with open(filename, 'wb') as file:
        dill.dump(obj, file)


def deserialize(filename):
    with open(filename, 'rb') as file:
        return dill.load(file)
