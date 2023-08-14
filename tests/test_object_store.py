import unittest
from pathlib import Path

import data

from researchrobot.config import get_config
from researchrobot.objectstore import ObjectStore


def datadir():

    return Path(data.__file__).parent


class Foobar:
    def __init__(self):
        self.x = "fooly"
        self.a = self

    def __eq__(self, other):
        return self.x == other.x


oc = [
    (True, b"B"),
    (False, b"B"),
    (0, b"i"),
    (1, b"i"),
    (1.2, b"f"),
    ("foo", b"s"),
    (b"bar", b"b"),
    ({"foo": "bar"}, b"J"),
    (["foo", "bar"], b"J"),
    (Foobar(), b"P"),
]


class TestObjectStore(unittest.TestCase):
    def setUp(self) -> None:
        conf_paths = [
            Path().home().joinpath(".robotcache.yaml"),
            datadir() / "robotcache.yaml",
        ]

        self.config = get_config(conf_paths)

    def _test_os(self, os: ObjectStore):
        print("test_os", os)

        # Start fresh
        for key in os:
            os.delete(key)

        for i, (o, tc) in enumerate(oc):
            key = f"{tc.decode('utf8')}-{i}"
            os[key] = o
            self.assertTrue(key in os)
            o2 = os[key]
            self.assertEqual(o, o2)

        self.assertEqual(len(list(os)), len(oc))

        for key in os:
            del os[key]

        self.assertEqual(len(list(os)), 0)

        with self.assertRaises(KeyError):
            o = os["missing"]

    def test_basic(self):

        cache_names = ["file", "local_minio", "local_redis", "spaces"]

        for name in cache_names:
            print(f"\n\n==================== {name} ======================")

            os = ObjectStore.new(
                name=name,
                bucket="testing.do.civicknowledge.com",
                prefix="unit-test",
                config=self.config,
            )

            self._test_os(os)


if __name__ == "__main__":
    unittest.main()
