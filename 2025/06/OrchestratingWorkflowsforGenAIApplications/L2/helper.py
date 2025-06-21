import os
import contextlib
                                                                                                                          
# Helper function to clean up longer logs
@contextlib.contextmanager
def suppress_output():
    """Suppress stdout and stderr including for child processes."""
    with open(os.devnull, 'w') as devnull:
        # Save actual file descriptors
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)

        # Redirect stdout/stderr to /dev/null
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            # Restore file descriptors
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)