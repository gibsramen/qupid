from .casematch import (CaseMatchOneToMany, CaseMatchOneToOne,
                        CaseMatchCollection)
from .qupid import match_by_single, match_by_multiple, shuffle


__version__ = "0.0.2"

__all__ = ["CaseMatchOneToMany", "CaseMatchOneToOne", "CaseMatchCollection",
           "match_by_single", "match_by_multiple", "shuffle"]
