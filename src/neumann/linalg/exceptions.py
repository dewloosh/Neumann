class VectorShapeMismatchError(Exception):
    """Mismatch in the shape of the inputs."""


class ArgumentError(Exception):
    """Invalid argument."""


class UfuncNotAllowedError(Exception):
    """Universal function not supported by this class."""
    

class LinalgOperationInputError(Exception):
    """Invalid input for this operation."""
    

class LinalgMissingInputError(Exception):
    """Invalid input for this operation."""