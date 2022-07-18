import re
from typing import Callable


def parse_help(args: str, arg_name: str) -> str:
    """Find the help string for an argument

    Search through the `Args` section of a function's
    doc string for the lines describing a particular
    argument. Returns the empty string if no description
    is found

    Args:
        args:
            The arguments section of a function docstring.
            Should be formatted like
            ```
            '''
            arg1:
                The description for arg1
            arg2:
                The description for arg 2 that
                spans multiple lines
            arg3:
                Another description
            '''
            ```
            With 8 spaces before each argument name
            and 12 before the lines of its description.
        arg_name:
            The name of the argument whose help string
            to search for
    Returns:
        The help string for the argument with leading
        spaces stripped for each line and newlines
        replaced by spaces
    """

    # TODO: more robustness on number of spaces
    arg_re = re.compile(rf"(?m)(\s){{8}}{arg_name}:$")

    doc_str, started = "", False
    for line in args.splitlines():
        # TODO: more robustness on spaces
        if arg_re.fullmatch(line):
            started = True
        elif not line.startswith(" " * 12) and started:
            break
        elif started:
            doc_str += " " + line.strip()
    return doc_str


def parse_doc(f: Callable):
    """Grab any documentation and argument help from a function"""

    # start by grabbing the function description
    # and any arguments that might have been
    # described in the docstring
    try:
        # split thet description and the args
        # by the expected argument section header
        doc, args = f.__doc__.split("Args:\n")
    except AttributeError:
        # raised if f doesn't have documentation
        doc, args = "", ""
    except ValueError:
        # raised if f only has a description but
        # no argument documentation. Set `args`
        # to the empty string
        doc, args = f.__doc__, ""
    else:
        # try to strip out any returns from the
        # arguments section by using the expected
        # returns header. If there are None, just
        # keep moving
        try:
            args, _ = args.split("Returns:\n")
        except ValueError:
            pass

    return doc, args
