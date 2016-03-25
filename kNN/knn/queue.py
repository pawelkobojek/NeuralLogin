import heapq

class PriorityQueue:
    def __init__(self):
        self._queue = []

    def push(self, priority, item):
        heapq.heappush(self._queue, (-priority, item))

    def pop(self):
        if len(self._queue) == 0:
            return None
        return heapq.heappop(self._queue)[-1]

    def peek_priority(self):
        return -self._queue[0][0]

    def size(self):
        return len(self._queue)
