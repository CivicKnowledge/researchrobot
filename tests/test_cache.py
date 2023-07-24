import os
import unittest

os.environ["MINIO_URL"] = "buzzkill.local:9000"

from researchrobot.cache import RedisCache, RobotCache, config


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


class TestBasic(unittest.TestCase):
    def setUp(self) -> None:

        self.rc = RobotCache("unit-test")
        print("minio url: ", config["MINIO_URL"])

    def test_conf(self):
        self.assertEqual(config["MINIO_URL"], "buzzkill.local:9000")

    def test_basic(self):
        rc = self.rc.sub("test-basic")

        for key, _ in rc:
            rc.delete(key)

        for i, (o, tc) in enumerate(oc):
            key = f"{tc.decode('utf8')}-{i}"
            rc[key] = o

            o2 = rc[key]
            self.assertEqual(o, o2)

    def test_redis_bytes(self):

        t = RedisCache(self.rc)

        # Check that the type codes are correct
        for o, tc in oc:
            tc_, b = t.to_bytes(o)
            self.assertEqual(tc, tc_)

        # Check that values can be converted to and from bytes
        for o, tc in oc:
            tc, b = t.to_bytes(o)
            o2 = t.from_bytes(tc + b)
            self.assertEqual(o, o2)

    def test_redis_kv(self):

        # Test key/values
        for i, (o, tc) in enumerate(oc):
            key = f"{tc.decode('utf8')}-{i}"
            self.rc.kv[key] = o
            o2 = self.rc.kv[key]
            self.assertEqual(o, o2)

    def test_redis_set(self):

        s = self.rc.set

        s.add(1)
        s.add(2)

        self.assertTrue(1 in s)
        self.assertTrue(2 in s)

        self.assertEqual([1, 2], list(sorted(s)))

        s.remove(1)

        self.assertFalse(1 in s)
        self.assertTrue(2 in s)

        print(list(s))


if __name__ == "__main__":
    unittest.main()
