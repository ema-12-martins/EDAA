class Node:
    def __init__(self, key):
        self.key = key
        self.degree = 0
        self.parent = None
        self.child = None
        self.mark = False
        self.left = self
        self.right = self

class FibonacciHeap:
    def __init__(self):
        self.min_node = None
        self.total_nodes = 0

    def insert(self, key):
        node = Node(key)
        if self.min_node is None:
            self.min_node = node
        else:
            self._add_node(node, self.min_node)
            if node.key < self.min_node.key:
                self.min_node = node
        self.total_nodes += 1

    def find_min(self):
        if self.min_node is None:
            return None
        return self.min_node.key

    def extract_min(self):
        z = self.min_node
        if z is not None:
            if z.child is not None:
                children = [x for x in self._iterate(z.child)]
                for child in children:
                    self._add_node(child, z)
                    child.parent = None
            self._remove_node(z)
            if z == z.right:
                self.min_node = None
            else:
                self.min_node = z.right
                self._consolidate()
            self.total_nodes -= 1
        return z.key if z else None

    def _add_node(self, node, root):
        node.left = root
        node.right = root.right
        root.right = node
        node.right.left = node

    def _remove_node(self, node):
        node.left.right = node.right
        node.right.left = node.left

    def _iterate(self, head):
        node = stop = head
        flag = False
        while True:
            if node == stop and flag is True:
                break
            elif node == stop:
                flag = True
            yield node
            node = node.right

    def _consolidate(self):
        A = [None] * self.total_nodes
        nodes = [x for x in self._iterate(self.min_node)]
        for w in nodes:
            x = w
            d = x.degree
            while A[d] is not None:
                y = A[d]
                if x.key > y.key:
                    x, y = y, x
                self._link(y, x)
                A[d] = None
                d += 1
            A[d] = x
        self.min_node = None
        for i in range(len(A)):
            if A[i] is not None:
                if self.min_node is None:
                    self.min_node = A[i]
                else:
                    self._add_node(A[i], self.min_node)
                    if A[i].key < self.min_node.key:
                        self.min_node = A[i]

    def _link(self, y, x):
        self._remove_node(y)
        y.left = y.right = y
        self._add_node(y, x.child if x.child else x)
        y.parent = x
        x.child = y
        x.degree += 1
        y.mark = False

# Example usage:
fib_heap = FibonacciHeap()
fib_heap.insert(10)
fib_heap.insert(2)
fib_heap.insert(15)

print("Minimum:", fib_heap.find_min())  # Output: Minimum: 2
print("Extracted Minimum:", fib_heap.extract_min())  # Output: Extracted Minimum: 2
print("New Minimum:", fib_heap.find_min())  # Output: New Minimum: 10