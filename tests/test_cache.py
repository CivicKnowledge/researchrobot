import os
import unittest

os.environ["MINIO_URL"] = "barker.local:9000"

from researchrobot.cache import RobotCache


class Foobar:
    def __init__(self):
        self.x = "fooly"
        self.a = self


class TestBasic(unittest.TestCase):
    def setUp(self) -> None:
        self.rc = RobotCache("unit-test")

    def test_basic(self):
        rc = self.rc.sub("test-basic")

        for key, _ in rc:
            rc.delete(key)

        rc["str"] = "str"
        rc["bytes"] = b"bytes"
        rc["dict"] = {"foo": "bar"}
        rc["list"] = ["foo", "bar"]
        rc["pickle"] = Foobar()

        self.assertEqual(rc["str"], "str")
        self.assertEqual(rc["bytes"], b"bytes")
        self.assertEqual(rc["dict"], {"foo": "bar"})
        self.assertEqual(rc["list"], ["foo", "bar"])
        self.assertEqual(rc["pickle"].x, "fooly")


if __name__ == "__main__":
    unittest.main()
