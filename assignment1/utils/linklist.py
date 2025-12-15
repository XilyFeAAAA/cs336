from dataclasses import dataclass
from typing import TypeVar, Generic, List


T = TypeVar("T")

@dataclass(eq=False)
class LinkNode(Generic[T]):
    pre = None
    nxt = None
    value: T




class LinkedList(Generic[T]):
    def __init__(self):
        self.length = -1
        self.head: LinkedNode[T] | None = None
        self.tail: LinkedNode[T] | None = None

    # 尾部插入
    def push_back(self, value) -> LinkNode[T]:
        node = LinkNode[T](value=value)
        if self.head is None:
            self.head = self.tail = node
        else:
            self.tail.nxt = node
            node.pre = self.tail
            self.tail = node
        return node

    def push_after(self, pre_node, value) -> LinkNode[T]:
        node = LinkNode[T](value=value)
        nxt_node = node.nxt
        node.pre = pre_node
        node.nxt = nxt_node
        pre_node.nxt = node
        nxt_node.pre = node
        return node

    def delete_node(self, node) -> None:
        if node.pre:
            node.pre.nxt = node.nxt
        else:
            self.head = node.nxt
        if node.nxt:
            node.nxt.pre = node.pre
        else:
            self.tail = node.pre
