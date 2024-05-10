from pymorphogen.core import ZStack
import numpy as np 

def test_z_stack():
    img = np.random.randint(0, 255, (100, 100, 10, 3), dtype=np.uint8)
    z_stack = ZStack(img)
    z_stack.show()


if __name__ == "__main__":
    test_z_stack()