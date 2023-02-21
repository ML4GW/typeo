import argparse
import inspect
import sys
import types
from collections import abc
from enum import Enum
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import typeo.actions as actions
from typeo.doc_utils import parse_doc, parse_help

if TYPE_CHECKING:
    try:
        from types import GenericAlias
    except ImportError:
        from typing import _GenericAlias as GenericAlias


_LIST_ORIGINS = (list, abc.Sequence, abc.Iterable)
_ARRAY_ORIGINS = _LIST_ORIGINS + (tuple,)
_DICT_ORIGINS = (dict, abc.Mapping)
_ANNOTATION = Union[type, "GenericAlias"]
_MAYBE_TYPE = Optional[type]


class CustomHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def _get_help_string(self, action):
        return argparse.ArgumentDefaultsHelpFormatter._get_help_string(
            self, action
        )


def _parse_union(param: inspect.Parameter) -> Tuple[type, _MAYBE_TYPE]:
    annotation = param.annotation

    try:
        # type_ will be the type that we pass to the
        # parser. second_type can either be `None` or
        # some array of type_
        type_, second_type = annotation.__args__
    except TypeError:
        raise ValueError(
            "Can't parse argument {} with annotation {} "
            "that is a Union of more than 2 types.".format(
                param.name, annotation
            )
        )

    try:
        origin = second_type.__origin__
    except AttributeError:
        # if the second argument to Union doesn't have
        # an origin, it's not array-like and so is
        # only allowed if it's None (which basically
        # corresponds to the `Optional` case)
        try:
            # see if the second type in the Union is NoneType
            is_none = isinstance(None, second_type)
        except TypeError:
            # annotation.__args__[1] is not a type that we
            # can check `None` as an instance of, so the
            # `isinstance` check above raises a TypeError.
            is_none = False

        if is_none:
            # if this second argument is None, i.e. we
            # have an Optional, the default must be None
            # otherwise there's no way for us to ever
            # be able to parse that `None` value from the
            # command line
            if param.default is not None:
                raise ValueError(
                    "Argument {} with Union of type {} and "
                    "NoneType must have a default of None".format(
                        param.name, type_
                    )
                )
            # we're done parsing now so return the two types
            return type_, None
        else:
            # this annotation isn't NoneType and doesn't have an
            # __origin__, so we can infer that it's not array-like
            # and we don't know how to parse this arg
            raise TypeError(
                "Arg {} has Union of types {} and {}".format(
                    param.name, type_, second_type
                )
            )

    # check if the type passed to the array-like
    # second argument to Union matches with the
    # type of the first argument to Union
    if origin in _LIST_ORIGINS:
        idx_to_check = 0
    elif origin in _DICT_ORIGINS:
        idx_to_check = 1
    elif origin is tuple:
        idx_to_check = None
    else:
        raise TypeError(
            "Arg {} has Union of type {} and type {} "
            "with unknown origin {}".format(
                param.name, type_, second_type, origin
            )
        )

    try:
        args = second_type.__args__
    except AttributeError:
        # in py3.9, generic aliases with no type
        # specified won't have __args__ at all,
        # so this is the same as being TypeVar
        # in py<3.9 and we're ok
        pass
    else:
        if idx_to_check is not None:
            args = [args[idx_to_check]]

        for arg in args:
            if arg not in (type_, Ellipsis) and not isinstance(arg, TypeVar):
                raise TypeError(
                    "Type argument {} passed to annotation {} of "
                    "parameter {} can't be parsed".format(
                        arg, annotation, param.name
                    )
                )
    return second_type, type_


def _get_origin_and_type(
    annotation: _ANNOTATION, type_: _MAYBE_TYPE = None
) -> Tuple[_MAYBE_TYPE, _MAYBE_TYPE]:
    """Utility for parsing the origin of an annotation

    Returns:
        If the annotation has an origin, this will be that origin.
            Otherwise it will be `None`
        If the annotation does not have an origin, this will
            be the annotation. Otherwise it will be `None`
    """

    try:
        return annotation.__origin__, type_
    except AttributeError:
        # annotation has no origin, so assume it's
        # a valid type on its own
        return None, annotation


def _parse_literal(annotation: _ANNOTATION):
    args = annotation.__args__
    type_ = type(args[0])
    if len(args) > 1:
        assert all([isinstance(i, type_) for i in args[1:]])

    if isinstance(args[0], Callable):
        type_ = abc.Callable
        args = [f"{i.__module__}.{i.__name__}" for i in args]
    return type_, args


def _is_untyped(args):
    # args being None indicates untyped lists and tuples in py3.9,
    # and args[0] being TypeVar indicates untyped lists in py3.8
    return args is None or isinstance(args[0], TypeVar)


def _parse_array_like(
    annotation: _ANNOTATION, origin: _MAYBE_TYPE, type_: type
) -> Tuple[type, Optional[Callable], Optional[List]]:
    """Grab types and expected actions for array-like annotations"""

    try:
        args = annotation.__args__
    except AttributeError:
        args = None

    if origin in _DICT_ORIGINS:
        action = actions.MappingAction
        if not _is_untyped(args):
            # make sure that the expected type
            # for the dictionary key is string
            # TODO: add kwarg for parsing non-str
            # dictionary keys
            assert args[0] is str

            # the type used to parse the values for
            # the dictionary will be the type passed
            # the parser action
            type_ = args[1]
    else:
        action = None
        try:
            if not _is_untyped(args):
                type_ = args[0]
        except IndexError:
            # untyped Tuples in py3.8 will have an empty __args__
            pass
        else:
            # for tuples make sure that everything
            # has the same type
            if origin is tuple and not _is_untyped(args):
                # TODO: use a custom action to verify the
                # number of arguments and map to a tuple
                try:
                    for arg in annotation.__args__[1:]:
                        if arg is not Ellipsis:
                            assert arg == type_
                except IndexError:
                    # if the Tuple only has one arg, we don't need
                    # to worry about checking everything else
                    pass

    # check to see if this array-like container
    # contains literals, in which case parse out
    # the type and choices expected by the literal
    try:
        is_literal = type_.__origin__ is Literal
    except (TypeError, AttributeError):
        choices = None
    else:
        if is_literal:
            type_, choices = _parse_literal(type_)
        else:
            choices = None

    return type_, action, choices


def _parse_container(
    annotation: _ANNOTATION, origin: type, kwargs: dict, type_: type
) -> _MAYBE_TYPE:
    """Make sure container-like arguments pass the right type to the parser

    For an annotation with an origin, do some checks on the
    origin to make sure that the type and action argparse
    uses to parse the argument is correct. If the annotation
    doesn't have an origin, returns `None`.

    Args:
        annotation:
            The annotation for the argument
        origin:
            The origin of the annotation, if it exists,
            otherwise `None`
        kwargs:
            The dictionary of keyword arguments to be
            used to add an argument to the parser
    """

    if origin in _ARRAY_ORIGINS + _DICT_ORIGINS:
        kwargs["nargs"] = "+"
        type_, action, choices = _parse_array_like(annotation, origin, type_)

        # check if the type contained in the array-like
        # annotation requires any kind of special action
        # or is a literal and so needs specific choices
        if action is not None:
            kwargs["action"] = action
        if choices is not None:
            kwargs["choices"] = choices

    elif origin is abc.Callable:
        type_ = origin
    elif origin is Literal:
        type_, choices = _parse_literal(annotation)
        kwargs["choices"] = choices
    else:
        # this is a type with some unknown origin
        raise TypeError(f"Can't help with arg of type {origin}")

    return type_


def _standardize_none_origin(type_):
    """
    Standardize py39 and py310 annotations to the format
    required to express them in py38
    """

    origin = None
    if type_ in _ARRAY_ORIGINS:
        # this will happen for untyped containers in py39+,
        # so just treat it like the case for py38
        origin, type_ = type_, None
    else:
        try:
            if isinstance(type_, types.UnionType):
                origin = Union
        except AttributeError:
            pass
    return origin, type_


def _is_kwargs(param: inspect.Parameter):
    return param.kind is inspect._ParameterKind.VAR_KEYWORD


def _is_args(param: inspect.Parameter):
    return param.kind is inspect._ParameterKind.VAR_POSITIONAL


def make_parser(
    f: Callable,
    parser: argparse.ArgumentParser,
    extends: Optional[Callable] = None,
) -> Dict[str, bool]:
    """Build an argument parser for a function

    Builds an `argparse.ArgumentParser` object by using
    the arguments to a function `f`, as well as their
    annotations and docstrings (for help printing).
    The type support for annotations is pretty limited
    and needs more documenting here, but for a better
    idea see the unit tests in `../tests/unit/test_scriptify.py`.

    Args:
        f:
            The function to construct a command line
            argument parser for
        parser:
            An existing parser to which to add arguments.
    Returns:
        A mapping from the names of any boolean arguments
            to the function to their default values, to be
            used for typeo config parsing.
    """

    doc, args = parse_doc(f)
    parameters = dict(inspect.signature(f).parameters)

    if extends is not None:
        extend_func, provided_args = extends
        extend_parameters = inspect.signature(extend_func).parameters

        for arg, param in extend_parameters.items():
            if arg not in provided_args and arg not in parameters:
                parameters[arg] = param

        _, extended_args = parse_doc(extend_func)
        args += "\n" + extended_args

    # now iterate through the arguments of f
    # and add them as options to the parser
    booleans = {}
    for name, param in parameters.items():
        if _is_args(param) or _is_kwargs(param):
            # This skips **kwargs style arguments, which we won't
            # be able to parse
            continue

        annotation = param.annotation
        kwargs = {}

        # check to see if the annotation represents
        # a type that can be used by the parser, or
        # represents some container that needs
        # further parsing
        origin, type_ = _get_origin_and_type(annotation)

        if origin is None:
            # do some standardization of things that are
            # allowable in py39 and py310 that require a
            # different syntax in py38
            origin, type_ = _standardize_none_origin(type_)

        # if the annotation can have multiple types,
        # figure out which type to pass to the parser
        if origin is Union:
            annotation, type_ = _parse_union(param)

            # check the chosen type again to
            # see if it's a container of some kind
            origin, type_ = _get_origin_and_type(annotation, type_)
            if origin in _ARRAY_ORIGINS:
                kwargs["action"] = actions.MaybeIterableAction

        if origin is not None:
            # if the annotation has some sort of origin, this
            # indicates a container type, so do some further
            # parsing of it to see what types of objects this
            # container is meant to contain so we can pass
            # those to the parser
            type_ = _parse_container(annotation, origin, kwargs, type_)

        # our last origin check to see if type_ is typing.Callable,
        # in which case the origin will be abc.Callable which
        # is the type that we want
        origin, type_ = _get_origin_and_type(type_)
        if origin is not None:
            type_ = origin

        # add the argument docstring to the parser help
        kwargs["help"] = parse_help(args, name)

        if type_ is bool:
            if param.default is inspect._empty:
                # if the argument is a boolean and doesn't
                # provide a default, assume that setting it
                # as a flag indicates a `True` status
                kwargs["action"] = "store_true"
                booleans[name] = False
            else:
                # otherwise set the action to be the
                # _opposite_ of whatever the default is
                # so that if it's not set, the default
                # becomes the values
                booleans[name] = param.default
                action = str(not param.default).lower()
                kwargs["action"] = f"store_{action}"
        else:
            kwargs["type"] = type_

            # args without default are required,
            # otherwise pass the default to the parser
            if param.default is inspect._empty:
                kwargs["required"] = True
            else:
                kwargs["default"] = param.default

            if type_ is abc.Callable:
                kwargs["action"] = actions.CallableAction
            elif type_ is not None and issubclass(type_, Enum):
                kwargs["action"] = actions.EnumAction
                kwargs["choices"] = [i.value for i in type_]

        # use dashes instead of underscores for argument names
        name = name.replace("_", "-")
        parser.add_argument(f"--{name}", **kwargs)
    return booleans, parameters


def _make_extra_args_parsers(
    parameters: List[inspect.Parameter],
    func_name: str,
    args: Optional[Callable] = None,
    kwargs: Optional[Callable] = None,
) -> Tuple[
    Optional[argparse.ArgumentParser], Optional[argparse.ArgumentParser]
]:
    if kwargs is None and args is None:
        return (None, None), (None, None)

    _has_kwargs = any([_is_kwargs(p) for p in parameters.values()])
    if not _has_kwargs:
        if kwargs is not None and args is None:
            postfix = f"function {kwargs.__name__}"
        elif kwargs is not None:
            postfix = f"functions {kwargs.__name__} and {args.__name__}"
        else:
            postfix = f"function {args.__name__}"
        raise ValueError(
            "{} has no **kwargs argument for "
            "passing args of {}".format(func_name, postfix)
        )

    parsers = []
    for func in [args, kwargs]:
        if func is None:
            parsers.append((None, None))
            continue

        def dummy():
            pass

        params = []
        signature = inspect.signature(func)
        for param in signature.parameters.values():
            if param.name not in parameters:
                params.append(param)
        dummy.__signature__ = inspect.Signature(parameters=params)

        parser = argparse.ArgumentParser(add_help=False, prog="dummy")
        make_parser(dummy, parser)
        parsers.append((parser, func.__name__))
    return tuple(parsers)


def _parse_remainder(
    remainder: List[str],
    kw: Dict,
    args_parser: Optional[argparse.ArgumentParser] = None,
    kwargs_parser: Optional[argparse.ArgumentParser] = None,
    args_func_name: Optional[str] = None,
    kwargs_func_name: Optional[str] = None,
):
    if args_parser is not None:
        try:
            args_parser.parse_args(remainder)
        except SystemExit as e:
            args_parser.error(
                "Error while parsing *args from function {}: "
                "{}".format(args_func_name, str(e))
            )
        kw["args"] = remainder

    if kwargs_parser is not None:
        try:
            kwgs = kwargs_parser.parse_args(remainder)
        except SystemExit as e:
            kwargs_parser.error(
                "Error while parsing **kwargs from function {}: "
                "{}".format(kwargs_func_name, str(e))
            )
        kw.update(vars(kwgs))


def _parse_subcommand(
    func_kwargs: Dict[str, Any],
    scriptify_kwargs: Dict[str, Callable],
    parameters: List[inspect.Parameter],
):
    # see if a subprogram was specified
    try:
        subcommand = func_kwargs.pop("_subcommand")
    except KeyError:
        subcommand = subkw = None
    else:
        # if we specified a subcommand, extract the arguments
        # that are specific to it. Convert its name
        # back to underscores first to grab the actual function
        subcommand = scriptify_kwargs[subcommand.replace("-", "_")]
        subparams = inspect.signature(subcommand).parameters
        subkw = {}
        for name, param in subparams.items():
            if _is_kwargs(param):
                continue
            elif name not in parameters:
                subkw[name] = func_kwargs.pop(name)
            else:
                subkw[name] = func_kwargs[name]

    return subcommand, subkw


def _execute_subcommand(
    subkw: Optional[Dict], subcommand: Optional[Callable], result: Any
):
    # run the subcommand if one was specified
    if subcommand is not None:
        # if the main function returned a dictionary,
        # pass it as kwargs to the subcommand
        if isinstance(result, dict):
            subkw.update(result)
        return subcommand(**subkw)
    return result


def _execute_extends(
    extend_func: Callable,
    provided_args: List[str],
    kwargs: Dict[str, Any],
    result: Dict[str, Any],
    func_name: str,
):
    if len(provided_args) == 1:
        kwargs[provided_args[0]] = result
    elif len(provided_args) != len(result):
        # TODO: check if result doesn't have a __len__
        raise ValueError(
            "Expected function {} to return {} arguments {} "
            "for use in function {}, but only found {}".format(
                func_name,
                len(provided_args),
                ", ".join(provided_args),
                extend_func.__name__,
                len(result),
            )
        )
    else:
        for arg, val in zip(provided_args, result):
            kwargs[arg] = val

    extend_params = inspect.signature(extend_func).parameters
    extend_kwargs = {}
    for arg, value in kwargs.items():
        if arg in extend_params:
            extend_kwargs[arg] = value

    return extend_func(**extend_kwargs)


def _make_wrapper(
    f: Callable,
    prog: Optional[str] = None,
    extends: Optional[Callable] = None,
    return_result: bool = False,
    args: Optional[Callable] = None,
    kwargs: Optional[Callable] = None,
    **other_kwargs,
) -> Callable:
    # start with a parent parser that will initially
    # try to parse a typeo toml config argument
    # let the downstream parser worry about handling help.
    # Don't add the --typeo argument to it yet though, since
    # this will need information about boolean arguments
    # extracted from the downstream parsers
    parent_parser = argparse.ArgumentParser(
        prog="config-parser", add_help=False, conflict_handler="resolve"
    )

    # now build a parser for the main function `f` which
    # inherits from this parser and can parse whatever
    # the config parser can't understand. The point of
    # inheritance is so that the `-h` flag will trigger
    # help from this parser and include the '--typeo' flag
    description, _ = parse_doc(f)
    parser = argparse.ArgumentParser(
        prog=prog or f.__name__,
        description=description.rstrip(),
        formatter_class=CustomHelpFormatter,
        parents=[parent_parser],
    )
    booleans, parameters = make_parser(f, parser, extends)

    # make parsers fo
    args, kwargs = _make_extra_args_parsers(
        parameters, f.__name__, args, kwargs
    )
    args_parser, args_func_name = args
    kwargs_parser, kwargs_func_name = kwargs

    # if we have subcommands, add subparsers for each
    # one of them with their own arguments
    if len(other_kwargs) > 0:
        subparsers = parser.add_subparsers(dest="_subcommand", required=True)
        for func_name, func in other_kwargs.items():
            description, _ = parse_doc(func)
            subparser = subparsers.add_parser(
                func_name.replace("_", "-"),
                description=description,
                formatter_class=CustomHelpFormatter,
            )

            bools, _ = make_parser(func, subparser)
            booleans.update(bools)

    # now add an argument for parsing a config file, using
    # info about booleans stripped from downstream parsers
    parent_parser.add_argument(
        "--typeo",
        bools=booleans,
        nargs="?",
        required=False,
        default=None,
        action=actions.TypeoTomlAction,
        help=(
            "Path to a typeo TOML config file of the form "
            "`path(:section)(:command)`, where `section` "
            "and `command` are optional. `path` can either be "
            "the path to a config file or to a directory with "
            "a `pyproject.toml` that will be used as the config. "
            "If left blank, a `pyproject.toml` file will be "
            "searched for in the current working directory. "
            "`section` specifies a subtable of the config in "
            "which to search for arguments, and `command` specifies "
            "a subcommand of the main function to execute, whose "
            "arguments are assumed to fall in a subtable of the config "
            "by that name."
        ),
    )

    # now build a wrapper for the function `f` that
    # parses from the command line if no arguments
    # are passed in, and otherwise just calls `f`
    # regularly
    @wraps(f)
    def wrapper(*args, **kw):
        if not len(args) == len(kw) == 0:
            # if any arguments at all were provided, run f normally
            return f(*args, **kw)

        config_args, remainder = parent_parser.parse_known_args()
        if config_args.typeo is not None:
            # TODO: what's the best way to have command line
            # arguments override those in the typeo config?
            if remainder:
                raise ValueError(
                    "Found additional arguments '{}' when passing "
                    "typeo config".format(remainder)
                )
            remainder = config_args.typeo

        kw, remainder = parser.parse_known_args(remainder)
        kw = vars(kw)
        if remainder and (args_parser is None and kwargs_parser is None):
            parser.error(
                "unrecognized arguments: {}".format(" ".join(remainder))
            )
        elif remainder:
            _parse_remainder(
                remainder,
                kw,
                args_parser,
                kwargs_parser,
                args_func_name,
                kwargs_func_name,
            )

        # see if a subprogram was specified
        subcommand, subkw = _parse_subcommand(kw, other_kwargs, parameters)

        # run the main function and potentially the subcommand
        result = f(*args, **kw)
        result = _execute_subcommand(subkw, subcommand, result)

        if extends is not None:
            extend_func, provided_args = extends
            result = _execute_extends(
                extend_func, provided_args, kw, result, f.__name__
            )

        if return_result:
            return result

    return wrapper


def scriptify(*args, **kwargs) -> Callable:
    """Function wrapper for passing command line args to functions

    Builds a command line parser for the arguments
    of a function so that if it is called without
    any arguments, its arguments will be attempted
    to be parsed from `sys.argv`.

    Usage:
        If your file `adder.py` looks like ::

            from typeo import scriptify


            @scriptify
            def f(a: int, other_number: int = 1) -> int:
                '''Adds two numbers together

                Longer description of the process of adding
                two numbers together.

                Args:
                    a:
                        The first number to add
                    other_number:
                        The other number to add whose description
                        inexplicably spans multiple lines
                '''

                print(a + other_number)


            if __name__ == "__main__":
                f()

        Then from the command line (note that underscores
        get replaced by dashes!) ::
            $ python adder.py --a 1 --other-number 2
            3
            $ python adder.py --a 4
            5
            $ python adder.py -h
            usage: f [-h] --a A [--other-number OTHER_NUMBER]

            Adds two numbers together

                Longer description of the process of adding
                two numbers together.

            optional arguments:
              -h, --help            show this help message and exit
              --a A                 The first number to add
              --other-number OTHER_NUMBER
                                    The other number to add whose description inexplicably spans multiple lines  # noqa

    Args:
        f: The function to expose via a command line parser
        prog:
            The name to assign to command line parser `prog`
            argument. If not provided, `f.__name__` will
            be used.
        extends:
            Tuple specifying a function `g` whose arguments `f`
            is meant to provide, as well as the names of
            the arguments of `g` that the outputs of `f` will be
            used to feed. If specified, all other arguments of
            `g` will be exposed to the command line and passed
            through straight-fowardly. In this case, `f` is
            executed first, then its outputs are passed to
            the corresponding inputs of `g`.
        return_result:
            Whether to return the output of any functions executed
            when `f` is called without any arguments. If left as
            `False`, `f()` will return None. This is the default
            behavior due to issues with conda raising an error
            on script-executed functions with return values, see
            https://github.com/ML4GW/BBHNet/issues/173
    """

    # the only argument is the function itself,
    # so just treat this like a simple wrapper
    if len(args) == 1 and isinstance(args[0], Callable):
        return _make_wrapper(args[0], **kwargs)
    else:
        # we provided arguments to typeo above the
        # decorated function, so wrap the wrapper
        # using the provided arguments

        @wraps(scriptify)
        def wrapperwrapper(f):
            return _make_wrapper(f, *args, **kwargs)

        return wrapperwrapper


def spoof(
    f: Callable,
    *args,
    filename: Optional[str] = None,
    script: Optional[str] = None,
    command: Optional[str] = None,
) -> dict:
    """Utility function for validating function arguments

    Returns as a dictionary the arguments passed to a function
    `f` if it were run via typeo either using explicit command
    line arguments or a config. If no arguments other than
    `f` are supplied, this would the equivalent of parsing
    arguments from a `pyproject.toml` with a `tool.typeo`
    section in the current working directory.

    Args:
        f:
            The function whose input arguments to inspect
        *args:
            Command line strings to parse using typeo. If
            specified, none of `filename`, `script`, or
            `command` should be specified.
        filename:
            Path to a type config file to parse. If left as
            `None`, equivalent to specifying a `pyproject.toml`
            in the current working directory.
        script:
            Subsection of the config file from which to
            parse arguments. If left as `None`, all arguments
            from the `typeo` section of the config will
            be parsed.
        command:
            Subcommand of the indicated config and script
            to parse arguments from. If left as `None`, no
            command arguments will be parsed.
    Returns:
        A dictionary mapping from the name of input arguments
            to `f` to their values inside `f`'s namespace.
    """

    def wrapper(**kwargs):
        return kwargs

    wrapper.__signature__ = inspect.signature(f)

    if len(args) > 0:
        if any([i is not None for i in [filename, script, command]]):
            raise ValueError(
                "Cannot specify argv if specifying any of "
                "'filename', 'script', or 'command'"
            )
    else:
        typeo_arg = ""
        if filename is not None:
            typeo_arg += filename
        if script is not None:
            typeo_arg += ":" + script
        if command is not None:
            if script is None:
                typeo_arg += ":"
            typeo_arg += ":" + command

        args = ["--typeo"]
        if typeo_arg:
            args.append(typeo_arg)

    argv = sys.argv
    sys.argv = [None] + list(args)
    kwargs = scriptify(wrapper, return_result=True)()
    sys.argv = argv
    return kwargs
