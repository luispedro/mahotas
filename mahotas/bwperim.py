# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# Copyright (C) 2012  Luis Pedro Coelho
# 
# License: MIT (see COPYING file)
import warnings
warnings.warn(
'''Use

from mahotas.labeled import bwperim
''', DeprecationWarning)


from .labeled import bwperim
__all__ = ['bwperim']

