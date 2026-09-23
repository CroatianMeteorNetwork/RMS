""" Shared helpers for command-line interfaces of RMS scripts.

This module is intentionally lightweight (stdlib-only at import time) so that it can be imported at the
top of any script without pulling in heavy dependencies.
"""

from __future__ import print_function, division, absolute_import


def addConfigArgument(parser, help_text=None):
    """ Register the standard -c/--config argument on an argparse parser.

    Unlike the legacy nargs=1 pattern, the value is stored as a plain string, so scripts can pass
    cml_args.config directly to loadConfig() without indexing.

    Arguments:
        parser: [argparse.ArgumentParser] Parser to add the argument to.

    Keyword arguments:
        help_text: [str] Custom help text. If None, a standard description is used.

    """

    if help_text is None:
        help_text = ("Path to a config file which will be used instead of the default one."
            " To load the .config file in the given data directory, use '.' as the value.")

    parser.add_argument('-c', '--config', metavar='CONFIG_PATH', type=str, help=help_text)


def loadConfig(cml_args_config, dir_path=None):
    """ Load the RMS config given the value of the --config argument.

    Accepts the value in any historical shape (None, a plain string, or the 1-element list produced by
    the legacy nargs=1 pattern) and normalizes it before handing it to
    RMS.ConfigReader.loadConfigFromDirectory, which expects a list.

    Arguments:
        cml_args_config: [None/str/list] Value of cml_args.config from argparse.

    Keyword arguments:
        dir_path: [str or list] Path to the working directory (or several). If None, the current
            directory is used.

    Return:
        config: [Config instance] Loaded config.

    """

    # Deferred import so importing this module stays cheap
    import RMS.ConfigReader as cr

    if dir_path is None:
        dir_path = '.'

    # Normalize the config argument to the list shape loadConfigFromDirectory expects
    if isinstance(cml_args_config, str):
        cml_args_config = [cml_args_config]

    return cr.loadConfigFromDirectory(cml_args_config, dir_path)
