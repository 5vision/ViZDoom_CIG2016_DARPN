import numpy as np
import scipy.signal
import cv2


def discount_cumsum(x, discount):
    x = np.array(x).astype(np.float32)
    # See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering
    # Here, we have y[t] - discount*y[t+1] = x[t]
    # or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    return scipy.signal.lfilter([1], [1, -discount], x[::-1], axis=0)[::-1].astype(np.float32)

def preprocess(image, width, height):
    image = cv2.resize(image[0], (width, height), interpolation=cv2.INTER_AREA)
    return image.reshape((1, 1, height, width)).astype(np.float32) / 255.0

def parse_action(a):
    """
    In Mode.SPECTATOR the last action taken from spectator.
    """
    if a[0]:
        return 0 # ATTACK
    if a[1] and a[3]:
        return 1 # TURN_LEFT and MOVE_FORWARD
    if a[2] and a[3]:
        return 2 # TURN_RIGHT and MOVE_FORWARD
    if a[1]:
        return 3 # TURN_LEFT
    if a[2]:
        return 4 # TURN_RIGHT
    if a[3]:
        return 5 # MOVE_FORWARD
    return 6 # NOP

def create_action(p, deterministic=False, rnd=None):
    c = np.cumsum(p)
    if rnd is None:
        rnd = np.random
    a = c.searchsorted(rnd.uniform(0, c[-1]))
    
    if deterministic and a == 0 and p[0] < 0.1:
        a = np.argmax(p)
    
    if a == 0:
        return a, [1,0,0,0] # ATTACK
    if a == 1:
        return a, [0,1,0,1] # TURN_LEFT and MOVE_FORWARD
    if a == 2:
        return a, [0,0,1,1] # TURN_RIGHT and MOVE_FORWARD
    if a == 3:
        return a, [0,1,0,0] # TURN_LEFT
    if a == 4:
        return a, [0,0,1,0] # TURN_RIGHT
    if a == 5:
        return a, [0,0,0,1] # MOVE_FORWARD
    return a, [0,0,0,0] # NOP
