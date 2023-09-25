"""Parsing and reconstructing the O*Net SOC taxonomy"""


def make_soc(major=0, minor=0, broad=0, detailed=0, onet=0):
    """Construct an SOC from parts"""

    if onet is None:
        return f"{int(major):02d}-{int(minor):02d}{int(broad):1d}{int(detailed):1d}"
    else:
        return f"{int(major):02d}-{int(minor):02d}{int(broad):1d}{int(detailed):1d}.{int(onet):02d}"

def parse_soc(soc, heirarchy=True):
    """Parse an SOC into dict parts.

    If heirarchy is true, each of the components is a full SOC, with '00' for the
    missing parts. If false, the components are just the numeric values of the parts.
    """

    major = soc[0:2]
    minor = soc[3:5]
    broad = soc[5]
    detailed = soc[6]

    if '.' in soc:
        _, onet = soc.split('.')
    else:
        onet = 0

    if heirarchy:
        return {
            'major': make_soc(major, onet=None),
            'minor': make_soc(major, minor, onet=None),
            'broad': make_soc(major, minor, broad, onet=None),
            'detailed': make_soc(major, minor, broad, detailed, onet=None),
            'soc': make_soc(major, minor, broad, detailed, onet),
        }
    else:
        return {
            'major': major,
            'minor': minor,
            'broad': broad,
            'detailed': detailed,
            'onet': onet
        }
