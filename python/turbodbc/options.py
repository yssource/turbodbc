from turbodbc_intern import Options

def make_options(read_buffer_size=None,
                 parameter_sets_to_buffer=None,
                 prefer_unicode=None,
                 use_async_io=None):
    options = Options()

    if not read_buffer_size is None:
        options.read_buffer_size = read_buffer_size

    if not parameter_sets_to_buffer is None:
        options.parameter_sets_to_buffer = parameter_sets_to_buffer

    if not prefer_unicode is None:
        options.prefer_unicode = prefer_unicode

    if not use_async_io is None:
        options.use_async_io = use_async_io

    return options
