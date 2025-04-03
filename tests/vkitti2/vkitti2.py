import unittest
from vkitti2 import VKitti2
import os


class TestVKitti2(unittest.TestCase):
    def test__set_dataroot_1(self):
        dataroot = "dataroot"
        location = "location"
        variation = "variation"
        vkitti2 = VKitti2(dataroot, location, variation)
        expected = (
            os.path.join(dataroot, location, variation),
            "location",
            "variation",
        )
        self.assertEqual(vkitti2._set_dataroot(dataroot, location, variation), expected)

    def test__set_dataroot_2(self):
        dataroot = "dataroot"
        location = "location"
        variation = "variation"
        dataroot = os.path.join(dataroot, location, variation)
        vkitti2 = VKitti2(dataroot)
        expected = (dataroot, "location", "variation")
        self.assertEqual(vkitti2._set_dataroot(dataroot, None, None), expected)

    def test__set_dataroot_3(self):
        vkitti2 = VKitti2("dataroot")
        self.assertRaises(
            ValueError, vkitti2._set_dataroot, "dataroot", None, "variation"
        )
        self.assertRaises(
            ValueError, vkitti2._set_dataroot, "dataroot", "location", None
        )
