def apply_init_window(u, y, init_window):
    if not init_window:
        return u, y
    return u[init_window:], y[init_window:]