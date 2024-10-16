from typing import List, Union, TypeVar

T = TypeVar('T')  # Generic type


def append_to_front(lst: List[T], elements: Union[T, List[T]]) -> None:
    """Appends either a single element or a list of elements to the front of the list."""
    if isinstance(elements, list):
        # Extend the list at the front using slicing for multiple elements
        lst[:0] = elements
    else:
        # Insert a single element at the front
        lst.insert(0, elements)
