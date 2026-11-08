"""Bounded process-local cache for trusted, previously validated CPU samples."""
from collections import OrderedDict
import torch


class TensorLRU:
    def __init__(self, budget_bytes):
        if budget_bytes < 0:
            raise ValueError('Negative cache budget')
        self.budget = budget_bytes
        self.bytes = 0
        self.items = OrderedDict()

    def get(self, key, load):
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key][0]
        value = load()
        if not self.budget:
            return value
        storages = {}
        for tensor in value.values():
            if torch.is_tensor(tensor):
                storage = tensor.untyped_storage()
                storages[storage.data_ptr()] = storage.nbytes()
        size = sum(storages.values())
        if size <= self.budget:
            while self.bytes + size > self.budget:
                _, (_, removed) = self.items.popitem(last=False)
                self.bytes -= removed
            self.items[key] = (value, size)
            self.bytes += size
        return value
